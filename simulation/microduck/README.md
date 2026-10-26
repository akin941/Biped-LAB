# Microduck (upstream model)

The official Microduck MJCF and meshes live in the pinned `microduck_rl` checkout
(`third_party/microduck_rl/src/mjlab_microduck/robot/microduck/`) and are never copied
here (meshes are CC BY-NC-SA 4.0, ADR-0004). Use `bipedlab.upstream.microduck_scene()`.

Scenes: `scene.xml` (groundcontact), `scene_walk.xml`, `scene_rollers.xml`,
`scene_allcollisions.xml`, `*_backlash.xml` variants. See `docs/research/microduck_rl.md`.
