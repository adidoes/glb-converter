# GLB to OBJ Converter

Simple Bun script for converting binary `.glb` files to Wavefront `.obj`.

```sh
bun run ./glb-to-obj.ts
```

With no arguments, it converts `.glb` files in the current folder. Existing `.obj` files are skipped.

```sh
bun run ./glb-to-obj.ts model.glb
bun run ./glb-to-obj.ts ./models
bun run ./glb-to-obj.ts --force
```

For `model.glb`, outputs are written beside the source file:

```text
model.obj
model.mtl
model_textures/
```

To build a standalone executable:

```sh
bun run build
```

The converter exports mesh positions, triangle faces, normals, OBJ-flipped UVs, node transforms, materials, embedded textures, and Draco-compressed glTF meshes.

Embedded WebP textures are converted to PNG for better OBJ/MTL importer compatibility. This currently uses the macOS `sips` command-line tool.

OBJ/MTL cannot represent every glTF feature, so skins, animations, cameras, sparse accessors, and non-triangle primitives are not exported. glTF PBR materials are mapped to a matte MTL subset with comments for metallic-roughness and occlusion textures.
