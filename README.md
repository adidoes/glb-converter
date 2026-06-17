# GLB to OBJ Converter

Simple Bun script for converting binary `.glb` mesh geometry to `.obj`.

```sh
bun run ./glb-to-obj.ts
```

With no arguments, it converts `.glb` files in the current folder and writes each `.obj` next to its source file with the same basename. Existing `.obj` files are skipped.

```sh
bun run ./glb-to-obj.ts model.glb
bun run ./glb-to-obj.ts ./models
bun run ./glb-to-obj.ts --force
```

To build a standalone executable:

```sh
bun run build
```

This first pass exports mesh positions, triangle faces, normals, UVs, and node transforms. It does not export materials, textures, skins, animations, cameras, sparse accessors, Draco-compressed meshes, or non-triangle primitives.
