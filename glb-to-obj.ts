#!/usr/bin/env bun

import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";

type Gltf = {
  scene?: number;
  scenes?: Array<{ nodes?: number[] }>;
  nodes?: GltfNode[];
  meshes?: GltfMesh[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: Array<{ uri?: string; byteLength?: number }>;
};

type GltfNode = {
  name?: string;
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
};

type GltfMesh = {
  name?: string;
  primitives?: GltfPrimitive[];
};

type GltfPrimitive = {
  mode?: number;
  attributes?: Record<string, number>;
  indices?: number;
  extensions?: Record<string, unknown>;
};

type GltfAccessor = {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  normalized?: boolean;
  sparse?: unknown;
};

type GltfBufferView = {
  buffer?: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
};

type Vec2 = [number, number];
type Vec3 = [number, number, number];
type Mat4 = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;
const MODE_TRIANGLES = 4;

const COMPONENT_SIZES = new Map<number, number>([
  [5120, 1],
  [5121, 1],
  [5122, 2],
  [5123, 2],
  [5125, 4],
  [5126, 4],
]);

const TYPE_COMPONENTS = new Map<string, number>([
  ["SCALAR", 1],
  ["VEC2", 2],
  ["VEC3", 3],
  ["VEC4", 4],
  ["MAT2", 4],
  ["MAT3", 9],
  ["MAT4", 16],
]);

type ObjParts = {
  lines: string[];
  vertexOffset: number;
  uvOffset: number;
  normalOffset: number;
};

function parseGlb(bytes: Uint8Array): { gltf: Gltf; bin: Uint8Array } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (view.byteLength < 20 || view.getUint32(0, true) !== GLB_MAGIC) {
    throw new Error("Not a binary .glb file.");
  }

  const version = view.getUint32(4, true);
  if (version !== 2) {
    throw new Error(`Unsupported GLB version ${version}; only GLB 2.0 is supported.`);
  }

  const declaredLength = view.getUint32(8, true);
  if (declaredLength > view.byteLength) {
    throw new Error("GLB file is truncated.");
  }

  let offset = 12;
  let json: Gltf | undefined;
  let bin: Uint8Array | undefined;

  while (offset + 8 <= declaredLength) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    const chunkEnd = chunkStart + chunkLength;

    if (chunkEnd > declaredLength) {
      throw new Error("GLB chunk extends past end of file.");
    }

    const chunk = bytes.subarray(chunkStart, chunkEnd);
    if (chunkType === CHUNK_JSON) {
      json = JSON.parse(new TextDecoder().decode(chunk).trim()) as Gltf;
    } else if (chunkType === CHUNK_BIN) {
      bin = chunk;
    }

    offset = chunkEnd;
  }

  if (!json) {
    throw new Error("GLB is missing its JSON chunk.");
  }
  if (!bin) {
    throw new Error("GLB is missing its binary geometry chunk.");
  }

  return { gltf: json, bin };
}

function identity(): Mat4 {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
}

function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Array(16).fill(0) as Mat4;
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      out[column * 4 + row] =
        a[0 * 4 + row] * b[column * 4 + 0] +
        a[1 * 4 + row] * b[column * 4 + 1] +
        a[2 * 4 + row] * b[column * 4 + 2] +
        a[3 * 4 + row] * b[column * 4 + 3];
    }
  }
  return out;
}

function nodeMatrix(node: GltfNode): Mat4 {
  if (node.matrix) {
    if (node.matrix.length !== 16) {
      throw new Error(`Node "${node.name ?? "(unnamed)"}" has an invalid matrix.`);
    }
    return node.matrix as Mat4;
  }

  const [tx, ty, tz] = node.translation ?? [0, 0, 0];
  const [x, y, z, w] = node.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale ?? [1, 1, 1];

  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;

  return [
    (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
    (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
    (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

function transformPoint(matrix: Mat4, point: Vec3): Vec3 {
  const [x, y, z] = point;
  return [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
  ];
}

function transformNormal(matrix: Mat4, normal: Vec3): Vec3 {
  const a00 = matrix[0];
  const a01 = matrix[4];
  const a02 = matrix[8];
  const a10 = matrix[1];
  const a11 = matrix[5];
  const a12 = matrix[9];
  const a20 = matrix[2];
  const a21 = matrix[6];
  const a22 = matrix[10];
  const det =
    a00 * (a11 * a22 - a12 * a21) -
    a01 * (a10 * a22 - a12 * a20) +
    a02 * (a10 * a21 - a11 * a20);

  const [x, y, z] = normal;
  let nx: number;
  let ny: number;
  let nz: number;

  if (Math.abs(det) < Number.EPSILON) {
    nx = a00 * x + a01 * y + a02 * z;
    ny = a10 * x + a11 * y + a12 * z;
    nz = a20 * x + a21 * y + a22 * z;
  } else {
    const invDet = 1 / det;
    const inv00 = (a11 * a22 - a12 * a21) * invDet;
    const inv01 = (a02 * a21 - a01 * a22) * invDet;
    const inv02 = (a01 * a12 - a02 * a11) * invDet;
    const inv10 = (a12 * a20 - a10 * a22) * invDet;
    const inv11 = (a00 * a22 - a02 * a20) * invDet;
    const inv12 = (a02 * a10 - a00 * a12) * invDet;
    const inv20 = (a10 * a21 - a11 * a20) * invDet;
    const inv21 = (a01 * a20 - a00 * a21) * invDet;
    const inv22 = (a00 * a11 - a01 * a10) * invDet;

    nx = inv00 * x + inv10 * y + inv20 * z;
    ny = inv01 * x + inv11 * y + inv21 * z;
    nz = inv02 * x + inv12 * y + inv22 * z;
  }

  const length = Math.hypot(nx, ny, nz) || 1;
  return [nx / length, ny / length, nz / length];
}

function readComponent(view: DataView, offset: number, componentType: number, normalized = false): number {
  switch (componentType) {
    case 5120: {
      const value = view.getInt8(offset);
      return normalized ? Math.max(value / 127, -1) : value;
    }
    case 5121: {
      const value = view.getUint8(offset);
      return normalized ? value / 255 : value;
    }
    case 5122: {
      const value = view.getInt16(offset, true);
      return normalized ? Math.max(value / 32767, -1) : value;
    }
    case 5123: {
      const value = view.getUint16(offset, true);
      return normalized ? value / 65535 : value;
    }
    case 5125:
      return view.getUint32(offset, true);
    case 5126:
      return view.getFloat32(offset, true);
    default:
      throw new Error(`Unsupported component type ${componentType}.`);
  }
}

function readAccessor(gltf: Gltf, bin: Uint8Array, accessorIndex: number): number[][] {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error(`Missing accessor ${accessorIndex}.`);
  }
  if (accessor.sparse) {
    throw new Error(`Sparse accessor ${accessorIndex} is not supported yet.`);
  }
  if (accessor.bufferView === undefined) {
    throw new Error(`Accessor ${accessorIndex} has no bufferView.`);
  }

  const bufferView = gltf.bufferViews?.[accessor.bufferView];
  if (!bufferView) {
    throw new Error(`Missing bufferView ${accessor.bufferView}.`);
  }
  if (bufferView.buffer !== undefined && bufferView.buffer !== 0) {
    throw new Error("External or multiple GLB buffers are not supported yet.");
  }

  const componentSize = COMPONENT_SIZES.get(accessor.componentType);
  const componentCount = TYPE_COMPONENTS.get(accessor.type);
  if (!componentSize || !componentCount) {
    throw new Error(`Unsupported accessor format ${accessor.type}/${accessor.componentType}.`);
  }

  const stride = bufferView.byteStride ?? componentSize * componentCount;
  const start = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const end = (bufferView.byteOffset ?? 0) + bufferView.byteLength;
  const view = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const values: number[][] = [];

  for (let row = 0; row < accessor.count; row++) {
    const rowOffset = start + row * stride;
    if (rowOffset + componentSize * componentCount > end) {
      throw new Error(`Accessor ${accessorIndex} reads past its bufferView.`);
    }

    const tuple: number[] = [];
    for (let component = 0; component < componentCount; component++) {
      tuple.push(readComponent(
        view,
        rowOffset + component * componentSize,
        accessor.componentType,
        accessor.normalized,
      ));
    }
    values.push(tuple);
  }

  return values;
}

function sanitizeName(name: string): string {
  return name.trim().replace(/\s+/g, "_").replace(/[^A-Za-z0-9_.-]/g, "_") || "mesh";
}

function formatNumber(value: number): string {
  if (Object.is(value, -0)) {
    return "0";
  }
  return Number.isInteger(value) ? String(value) : Number(value.toFixed(8)).toString();
}

function convertPrimitive(
  gltf: Gltf,
  bin: Uint8Array,
  primitive: GltfPrimitive,
  matrix: Mat4,
  name: string,
  obj: ObjParts,
): void {
  if (primitive.extensions?.KHR_draco_mesh_compression) {
    throw new Error(`${name} uses Draco compression, which is not supported by this simple converter.`);
  }

  const mode = primitive.mode ?? MODE_TRIANGLES;
  if (mode !== MODE_TRIANGLES) {
    throw new Error(`${name} uses primitive mode ${mode}; only TRIANGLES are supported.`);
  }

  const positionAccessor = primitive.attributes?.POSITION;
  if (positionAccessor === undefined) {
    return;
  }

  const positions = readAccessor(gltf, bin, positionAccessor) as Vec3[];
  const normals = primitive.attributes?.NORMAL === undefined
    ? undefined
    : readAccessor(gltf, bin, primitive.attributes.NORMAL) as Vec3[];
  const uvs = primitive.attributes?.TEXCOORD_0 === undefined
    ? undefined
    : readAccessor(gltf, bin, primitive.attributes.TEXCOORD_0) as Vec2[];

  const rawIndices = primitive.indices === undefined
    ? positions.map((_, index) => index)
    : readAccessor(gltf, bin, primitive.indices).map(([index]) => index);

  if (rawIndices.length % 3 !== 0) {
    throw new Error(`${name} does not contain a whole number of triangles.`);
  }

  obj.lines.push(`o ${sanitizeName(name)}`);

  for (const position of positions) {
    const [x, y, z] = transformPoint(matrix, position);
    obj.lines.push(`v ${formatNumber(x)} ${formatNumber(y)} ${formatNumber(z)}`);
  }

  if (uvs) {
    for (const [u, v] of uvs) {
      obj.lines.push(`vt ${formatNumber(u)} ${formatNumber(v)}`);
    }
  }

  if (normals) {
    for (const normal of normals) {
      const [x, y, z] = transformNormal(matrix, normal);
      obj.lines.push(`vn ${formatNumber(x)} ${formatNumber(y)} ${formatNumber(z)}`);
    }
  }

  const vertexRef = (index: number): string => {
    const v = obj.vertexOffset + index + 1;
    const vt = uvs ? obj.uvOffset + index + 1 : "";
    const vn = normals ? obj.normalOffset + index + 1 : "";

    if (uvs && normals) {
      return `${v}/${vt}/${vn}`;
    }
    if (uvs) {
      return `${v}/${vt}`;
    }
    if (normals) {
      return `${v}//${vn}`;
    }
    return String(v);
  };

  for (let i = 0; i < rawIndices.length; i += 3) {
    obj.lines.push(`f ${vertexRef(rawIndices[i])} ${vertexRef(rawIndices[i + 1])} ${vertexRef(rawIndices[i + 2])}`);
  }

  obj.vertexOffset += positions.length;
  obj.uvOffset += uvs?.length ?? 0;
  obj.normalOffset += normals?.length ?? 0;
}

function convertGlbToObj(source: string): string {
  const bytes = readFileSync(source);
  const { gltf, bin } = parseGlb(bytes);
  const sceneIndex = gltf.scene ?? 0;
  const scene = gltf.scenes?.[sceneIndex];
  if (!scene) {
    throw new Error(`Missing scene ${sceneIndex}.`);
  }

  if (gltf.buffers?.[0]?.uri) {
    throw new Error("GLB files with external buffer URIs are not supported by this simple converter.");
  }

  const obj: ObjParts = {
    lines: [
      `# Converted from ${basename(source)} by glb-to-obj.ts`,
      "# Materials, skins, animations, cameras, and textures are not exported.",
    ],
    vertexOffset: 0,
    uvOffset: 0,
    normalOffset: 0,
  };

  const visitNode = (nodeIndex: number, parentMatrix: Mat4, path: string): void => {
    const node = gltf.nodes?.[nodeIndex];
    if (!node) {
      throw new Error(`Missing node ${nodeIndex}.`);
    }

    const nodeName = node.name ?? `node_${nodeIndex}`;
    const worldMatrix = multiply(parentMatrix, nodeMatrix(node));
    const nodePath = path ? `${path}_${nodeName}` : nodeName;

    if (node.mesh !== undefined) {
      const mesh = gltf.meshes?.[node.mesh];
      if (!mesh) {
        throw new Error(`Missing mesh ${node.mesh}.`);
      }

      mesh.primitives?.forEach((primitive, primitiveIndex) => {
        const meshName = mesh.name ?? `mesh_${node.mesh}`;
        convertPrimitive(gltf, bin, primitive, worldMatrix, `${nodePath}_${meshName}_${primitiveIndex}`, obj);
      });
    }

    for (const childIndex of node.children ?? []) {
      visitNode(childIndex, worldMatrix, nodePath);
    }
  };

  for (const nodeIndex of scene.nodes ?? []) {
    visitNode(nodeIndex, identity(), "");
  }

  if (obj.vertexOffset === 0) {
    throw new Error("No mesh geometry was found.");
  }

  return `${obj.lines.join("\n")}\n`;
}

function isGlbFile(path: string): boolean {
  return statSync(path).isFile() && extname(path).toLowerCase() === ".glb";
}

function collectGlbFiles(paths: string[]): string[] {
  const files: string[] = [];

  for (const path of paths) {
    const resolved = resolve(path);
    if (!existsSync(resolved)) {
      console.warn(`warn: ${path} does not exist; skipping`);
      continue;
    }

    const stats = statSync(resolved);
    if (stats.isDirectory()) {
      for (const entry of readdirSync(resolved)) {
        const candidate = join(resolved, entry);
        if (isGlbFile(candidate)) {
          files.push(candidate);
        }
      }
    } else if (isGlbFile(resolved)) {
      files.push(resolved);
    } else {
      console.warn(`warn: ${path} is not a .glb file or directory; skipping`);
    }
  }

  return [...new Set(files)].sort((a, b) => a.localeCompare(b));
}

function outputPathFor(source: string): string {
  return join(dirname(source), `${basename(source, extname(source))}.obj`);
}

function printHelp(): void {
  console.log(`Usage:
  bun run glb-to-obj.ts [--force] [file-or-folder ...]

Defaults:
  With no paths, converts .glb files in the current folder.
  Existing .obj files are skipped unless --force is set.

Examples:
  bun run glb-to-obj.ts
  bun run glb-to-obj.ts model.glb
  bun build --compile ./glb-to-obj.ts --outfile glb-to-obj`);
}

function main(): void {
  const args = process.argv.slice(2);
  const force = args.includes("--force");
  const help = args.includes("--help") || args.includes("-h");
  const paths = args.filter((arg) => arg !== "--force" && arg !== "--help" && arg !== "-h");

  if (help) {
    printHelp();
    return;
  }

  const files = collectGlbFiles(paths.length > 0 ? paths : [process.cwd()]);
  if (files.length === 0) {
    console.log("No .glb files found.");
    return;
  }

  let converted = 0;
  let skipped = 0;
  let failed = 0;

  for (const source of files) {
    const target = outputPathFor(source);
    if (!force && existsSync(target)) {
      console.log(`skip: ${source} -> ${target} already exists`);
      skipped++;
      continue;
    }

    try {
      const obj = convertGlbToObj(source);
      writeFileSync(target, obj);
      console.log(`ok: ${source} -> ${target}`);
      converted++;
    } catch (error) {
      failed++;
      const message = error instanceof Error ? error.message : String(error);
      console.error(`fail: ${source}: ${message}`);
    }
  }

  console.log(`Done. converted=${converted} skipped=${skipped} failed=${failed}`);
  if (failed > 0) {
    process.exitCode = 1;
  }
}

main();
