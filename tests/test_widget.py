from napari_clipping_planes_controls.controls_widget import (
    ClippingPlanesControls,
)


def test_create_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widg = viewer.window.add_plugin_dock_widget(
        'napari-clipping-planes-controls'
    )
    assert isinstance(widg[1], ClippingPlanesControls)
