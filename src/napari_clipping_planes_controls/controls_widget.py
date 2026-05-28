import warnings

import napari
import numpy as np
from qtpy.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.transform import Rotation
from superqt import QLabeledDoubleSlider, QToggleSwitch
from vispy.scene import ArcballCamera, Box, InfiniteLine, SceneCanvas
from vispy.util.quaternion import Quaternion
from vispy.visuals.transforms import MatrixTransform, STTransform


class ClippingPlanesSideViewControls(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        # general layout
        lay = QVBoxLayout()
        self.setLayout(lay)

        # TODO: update bgcolor in sync with main canvas
        self.canvas = SceneCanvas(keys=None, size=(300, 300), vsync=True)

        self.info = QLabel(
            'This is a side view of the main napari canvas.\n'
            'Select a layer in the layerlist, then \n'
            'click/drag below to move its clipping planes.'
        )
        self.planes_locked = QToggleSwitch('Lock planes.')

        lay.addWidget(self.info)
        lay.addWidget(self.planes_locked)
        lay.addWidget(self.canvas.native)

        self.view = self.canvas.central_widget.add_view()
        self.box = None
        self._center = np.zeros(3)
        self._locked = False

        self.view.camera = ArcballCamera(fov=0)
        self.view.camera.interactive = False
        self.camera_model = napari.components.Camera()

        self.lines = [
            InfiniteLine(pos=0, color=(1, 0, 1, 1), parent=self.view),
            InfiniteLine(pos=0, color=(1, 0, 0, 1), parent=self.view),
        ]
        self._offsets = [-1e10, 1e10]
        for line in self.lines:
            line.transform = STTransform()

        self.viewer.dims.events.range.connect(self._on_extent_change)
        self.viewer.dims.events.ndisplay.connect(self._update_lock_and_view)
        self.viewer.camera.events.connect(self._on_camera_change)
        self.viewer.layers.selection.events.connect(self._update_lock_and_view)
        self.canvas.events.mouse_move.connect(self._on_mouse)
        self.canvas.events.mouse_press.connect(self._on_mouse)
        self.canvas.events.resize.connect(self._on_size_change)
        self.planes_locked.toggled.connect(self._update_lock_and_view)

        self._update_lock()
        self._on_extent_change()

    def _on_extent_change(self):
        self._update_lock()
        if self._locked:
            return
        displayed = self.viewer.dims.displayed

        if self.box is not None:
            self.box.parent = None

        extents = self.viewer.layers.get_extent(
            self.viewer.layers.selection
        ).world[:, displayed]
        sizes = extents[1] - extents[0]
        if len(sizes) == 2:
            # 3D view but only 2D layers, add z
            sizes = np.pad(sizes, (1, 0))
        self.box = Box(
            *sizes[::-1],  # swap for vispy
            face_colors=np.repeat([[0, 1, 1]], 12, axis=0),
            edge_color='black',
            parent=self.view.scene,
        )

        self._center = sizes / 2

        self.view.camera.set_range(margin=0.2)

        self._update_depth()
        self._on_camera_change()

    def _update_depth(self):
        # same code as QtViewer._update_camera_depth()
        extent = self.viewer.layers.extent
        extent_all = extent.world[1] - extent.world[0] + extent.step
        extent_displayed = extent_all[list(self.viewer.dims.displayed)]
        diameter = np.linalg.norm(extent_displayed)
        self.view.camera.depth_value = 128 * diameter

    def _update_lock(self):
        self._locked = (
            self.viewer.dims.ndisplay == 2
            or not self.viewer.layers.selection
            or self.planes_locked.isChecked()
        )

    def _update_lock_and_view(self):
        self._update_lock()
        self.canvas.native.setVisible(not self._locked)
        self.view.visible = not self._locked
        if not self._locked:
            self._on_extent_change()
            self._update_planes()

    def _on_size_change(self, event=None):
        X = self.canvas.size[0] / 2
        self.lines[0].transform.translate = (X + self._offsets[0], 0, 0)
        self.lines[1].transform.translate = (X + self._offsets[1], 0, 0)
        self._on_camera_change()

    def _on_camera_change(self):
        self._update_lock()
        if self._locked or self.box is None:
            return

        # I don't understand why this is needed to align to the napaari view,
        # but it is for some reason :/ TODO figure out why
        side_view = Rotation.from_euler('x', 90, degrees=True)
        mat = np.eye(4)
        mat[:3, :3] = side_view.as_matrix()
        self.box.transform = MatrixTransform(mat)

        # rotate 90 degrees so we see from the side
        up = self.viewer.camera.up_direction
        view = self.viewer.camera.view_direction
        side = Rotation.from_rotvec(np.array(up) * 90, degrees=True).apply(
            view
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='gimbal lock')
            self.camera_model.set_view_direction(side, up)

        # --------------------------------------------------------
        # this section is copied from VispyCamera logic
        # flip handedness so the rotation is always righthanded even with axis flipping
        angles = self.camera_model.angles * np.where(
            self.camera_model._vispy_flipped_axes(ndisplay=3), -1, 1
        )
        # undo vispy quirks (rotation of 90 digrees and lefthanded y axis)
        angles = (np.array(angles) * (1, -1, 1)) + (0, 0, 90)
        # see #8281 for why this is yzx. In short: longstanding vispy bug.
        rotation = Rotation.from_euler('yzx', angles, degrees=True)

        # Create and set quaternion
        q = Quaternion(*rotation.as_quat(scalar_first=True))
        self.view.camera._quaternion = q
        # --------------------------------------------------------

        VISPY_DEFAULT_ORIENTATION_3D = ('right', 'down', 'away')  # xyz
        self.view.camera.flip = tuple(
            int(ori != default_ori)
            for ori, default_ori in zip(
                self.viewer.camera.orientation[::-1],
                VISPY_DEFAULT_ORIENTATION_3D,
                strict=True,
            )
        )

        self.view.camera.view_changed()
        self.view.camera.center = (self.viewer.camera.center - self._center)[
            ::-1
        ]

        self._update_planes()

    def _on_mouse(self, event=None):
        self._update_lock()
        if self._locked:
            return
        if self.viewer.layers.selection and (
            event.type == 'mouse_press' or event.button == 1
        ):
            X = self.canvas.size[0]
            x = event.pos[0]
            offset = x - X / 2
            if offset <= 0:
                self.lines[0].transform.translate = (x, 0, 0)
                self._offsets[0] = offset
            else:
                self.lines[1].transform.translate = (x, 0, 0)
                self._offsets[1] = offset

            self._update_planes()

    def _update_planes(self):
        self._update_lock()
        if self._locked:
            return
        planes_world = []
        for offset in self._offsets:
            zoom = np.min(
                np.array(self.canvas.size) / self.view.camera.scale_factor
            )
            data_offset = offset / zoom
            offset_direction = np.array(self.viewer.camera.view_direction)
            offset_position = (
                self.viewer.camera.center + offset_direction * data_offset
            )
            planes_world.append([offset_position, offset_direction])

        # flip normal for second plane
        planes_world[1][1] *= -1

        # set plane data to each layer
        displayed = list(self.viewer.dims.displayed)
        for layer in self.viewer.layers.selection:
            planes = []
            for plane in planes_world:
                world_pos_full = np.zeros(self.viewer.dims.ndim)
                world_pos_full[displayed] = plane[0]

                world_norm_full = np.zeros(self.viewer.dims.ndim)
                world_norm_full[displayed] = plane[1]

                planes.append(
                    {
                        'position': world_pos_full,
                        'normal': world_norm_full,
                        'enabled': True,
                    }
                )
            layer.experimental_clipping_planes = planes


class ClippingPlanesSliderControls(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        # general layout
        lay = QVBoxLayout()
        self.setLayout(lay)

        self.info = QLabel(
            'This widget provides slider controls for clipping planes.\n'
            'Select a layer in the layerlist, then use the +/- button\n'
            'to add/remove clipping planes, and the respective sliders\n'
            'to change their position and direction.\n'
            'Slider position is relative to the bounding sphere\n'
            'of the selected layer extent.'
        )

        btns = QHBoxLayout()
        self.add_btn = QPushButton('+')
        self.remove_btn = QPushButton('-')
        btns.addWidget(self.add_btn)
        btns.addWidget(self.remove_btn)

        self.sliders_layout = QVBoxLayout()

        lay.addWidget(self.info)
        lay.addLayout(btns)
        lay.addLayout(self.sliders_layout)

        self.slider_map = {}
        self.plane_directions = {}
        self.selection = None

        self.viewer.layers.selection.events.active.connect(
            self._on_selection_changed
        )
        self.add_btn.pressed.connect(self._add_plane)
        self.remove_btn.pressed.connect(self._remove_plane)

        self._on_selection_changed()

    def _on_selection_changed(self):
        active = self.viewer.layers.selection.active
        if active is not None:
            active.experimental_clipping_planes.events.inserted.connect(
                self._update_sliders
            )
            active.experimental_clipping_planes.events.removed.connect(
                self._update_sliders
            )
        if self.selection is not None:
            self.selection.experimental_clipping_planes.events.inserted.disconnect(
                self._update_sliders
            )
            self.selection.experimental_clipping_planes.events.removed.disconnect(
                self._update_sliders
            )
        self.selection = active

        self._update_sliders()

    def _add_plane(self):
        if self.selection is None:
            return
        self.selection.experimental_clipping_planes.add_plane()

    def _remove_plane(self):
        if self.selection is None:
            return
        self.selection.experimental_clipping_planes.pop()

    def _update_sliders(self):
        _clear_layout(self.sliders_layout)
        self.slider_map.clear()

        if self.selection is None:
            return

        for i, plane in enumerate(self.selection.experimental_clipping_planes):
            sliders = self._make_plane_sliders(plane)
            self.slider_map[plane] = sliders

            self.sliders_layout.addWidget(QLabel(f'Plane {i}:'))
            lay = QFormLayout()
            for label, slider in sliders.items():
                lay.addRow(label, slider)
            self.sliders_layout.addLayout(lay)

    def _get_bounding_sphere(self):
        displayed = self.viewer.dims.displayed
        extents = self.viewer.layers.get_extent(
            self.viewer.layers.selection
        ).world[:, displayed]

        sizes = extents[1] - extents[0]
        if len(sizes) == 2:
            # 3D view but only 2D layers, add z
            sizes = np.pad(sizes, (1, 0))
        sphere_center = np.mean(extents, axis=0)
        sphere_radius = 0.5 * np.sqrt(np.sum(sizes**2))

        return sphere_center, sphere_radius

    def _make_plane_sliders(self, plane):
        sliders = {}
        angles = _normal_to_angles(plane.normal)
        for name, angle in zip(('rot', 'tilt', 'psi'), angles, strict=True):
            slider = QLabeledDoubleSlider()
            slider.setRange(0, 180)
            slider.setValue(angle)
            sliders[name] = slider

        slider = QLabeledDoubleSlider()
        slider.setRange(-1, 1)
        sphere_center, sphere_radius = self._get_bounding_sphere()
        pos = plane.position - sphere_center
        self.plane_directions[plane] = pos / np.linalg.norm(pos)
        pos_rel = np.linalg.norm(pos) / sphere_radius
        slider.setValue(pos_rel)
        sliders['pos'] = slider

        callback = self._plane_update_callback(plane)
        for slider in sliders.values():
            slider.valueChanged.connect(callback)

        return sliders

    def _plane_update_callback(self, plane):
        def plane_callback(value):
            sliders = self.slider_map[plane]
            angles = np.array([s.value() for s in sliders.values()][:-1])
            normal = _angles_to_normal(angles)
            if not np.allclose(normal, plane.normal):
                plane.normal = normal

            sphere_center, sphere_radius = self._get_bounding_sphere()
            rel_pos = sliders['pos'].value()
            pos = (
                sphere_center
                + rel_pos * sphere_radius * 2 * self.plane_directions[plane]
            )
            if not np.allclose(pos, plane.position):
                plane.position = pos

        return plane_callback


def _normal_to_angles(normal):
    rot = Rotation.align_vectors([normal], [[0, 0, 1]])[0]
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='gimbal lock')
        return rot.as_euler('xyz', degrees=True)


def _angles_to_normal(angles):
    rot = Rotation.from_euler('xyz', angles, degrees=True)
    return rot.apply([0, 0, 1])


def _clear_layout(layout):
    for _ in range(layout.count()):
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        else:
            child_layout = item.layout()
            if child_layout is not None:
                _clear_layout(child_layout)
