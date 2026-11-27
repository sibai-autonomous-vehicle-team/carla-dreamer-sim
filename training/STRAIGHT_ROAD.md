Controlled road preview and training
===================================

Use `--road-environment straight` in a fresh `training.train_online_hj` or
`training.train_state_hj` run. Town runs retain their original scene, camera,
weather and vehicle defaults. `--straight-road-length` defaults to 1000 m.

The OpenDRIVE map supplies flat road collision geometry and the existing right
lane route. Separate noncolliding CARLA meshes render 7.2 m of asphalt, a dashed
center line and flat green grass verges. There are no junctions, buildings or
other traffic. Vehicle speeds, gaps and lead braking still use the existing
training/evaluation samplers and controls.

The straight environment uses `vehicle.audi.tt` for both cars, configurable with
`--straight-vehicle`. Its integrated headlights/tail lamps render at night;
the previous Audi A2 did not produce usable lights in the preview. Vehicle
appearance and dynamics therefore differ from the old town experiment.

All conditions share the native 224x224 RGB sensor, 120-degree FOV, x=1.5 m,
z=2 m mounting and histogram exposure range 4..16 (the packaged sensor defaults
were 10..12). Adaptation speeds are 5 in both directions, with exposure
compensation -0.7. Bloom, lens flare and motion blur are disabled; f/8 and a
10 m focus distance reduce depth-of-field blur. All nine conditions use these
same camera settings. Sunset/night/rain use
position and low-beam lights; brake lamps follow actual braking commands. No
street lighting or image enhancement is added. The spectator follows the ego
in offscreen mode too, to keep precipitation effects near the active camera.

`training/straight_weather.py` defines explicit, separate sunny/cloudy/rainy
profiles and sun positions for this environment. The old shared weather
function and existing town experiments are unchanged. Rain uses precipitation
90, deposits 80 and wetness 100. Weather does not change tire friction here.
The reused prop mesh's normal and packed material maps are replaced with flat
normals and a rough, nonmetallic, nonemissive surface. Rain can still create wet
reflections, and headlights can still clip bright pixels near the vehicle.

Capture all nine matched static views on a separate idle server:

```bash
.venv/bin/python -m training.preview_straight_road \
  --outdir training/_local/road_preview --port 2950 --start-server
```

Override `--server` to point to your CARLA 0.9.15 installation. The preview
uses a 5 m bumper gap and freezes the cars for matched geometry; its stationary
history is not training data. It saves individual raw PNGs, four consecutive
history frames per condition, a contact sheet and a manifest of actual weather,
vehicle poses, frame IDs and lighting. No resizing or post-capture enhancement
is applied. Preview vehicle override: `--blueprint vehicle.audi.tt`.
