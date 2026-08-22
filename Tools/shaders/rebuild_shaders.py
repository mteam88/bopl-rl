#!/usr/bin/env python3
"""Rebuild original Bopl Battle shaders from the game's serialized shader data.

For each shader in the game data:
  1. Read m_ParsedForm -> exact ShaderLab properties, tags, render state.
  2. Decompress the program blob -> original GLSL (HLSLcc output, GLCore32 preferred).
  3. Translate the GLSL to CG/HLSL (mechanical, reviewed).
  4. Emit a complete .shader file.

Output goes to a staging dir for review; apply separately (GUIDs live in .meta).
"""
import re
import sys
import glob
import pathlib
import UnityPy
from UnityPy.export.ShaderConverter import ShaderProgram, CompressionHelper, ShaderGpuProgramType
from UnityPy.streams import EndianBinaryReader

GAME_DATA = '/home/mte/Documents/Coding/bopl-re/game/BoplBattle_Data'
OUT_DIR = pathlib.Path('/home/mte/Documents/Coding/bopl-re/original-shaders')

GL_PREFERENCE = [
    ShaderGpuProgramType.kShaderGpuProgramGLCore32,
    ShaderGpuProgramType.kShaderGpuProgramGLCore41,
    ShaderGpuProgramType.kShaderGpuProgramGLCore43,
    ShaderGpuProgramType.kShaderGpuProgramGLES31,
    ShaderGpuProgramType.kShaderGpuProgramGLES31AEP,
    ShaderGpuProgramType.kShaderGpuProgramGLES3,
    ShaderGpuProgramType.kShaderGpuProgramGLLegacy,
]

BLEND_FACTORS = {0: 'Zero', 1: 'One', 2: 'DstColor', 3: 'SrcColor', 4: 'OneMinusDstColor',
                 5: 'SrcAlpha', 6: 'OneMinusSrcColor', 7: 'DstAlpha', 8: 'OneMinusDstAlpha',
                 9: 'SrcAlphaSaturate', 10: 'OneMinusSrcAlpha'}
BLEND_OPS = {0: 'Add', 1: 'Sub', 2: 'RevSub', 3: 'Min', 4: 'Max', 5: 'LogicalClear',
             6: 'LogicalSet', 7: 'LogicalCopy', 8: 'LogicalCopyInverted', 9: 'LogicalNoop',
             10: 'LogicalInvert', 11: 'LogicalAnd', 12: 'LogicalNand', 13: 'LogicalOr',
             14: 'LogicalNor', 15: 'LogicalXor', 16: 'LogicalEquiv', 17: 'LogicalAndReverse',
             18: 'LogicalAndInverted', 19: 'LogicalOrReverse', 20: 'LogicalOrInverted'}
ZTEST = {0: 'Off', 1: 'Never', 2: 'Less', 3: 'Equal', 4: 'LEqual', 5: 'Greater',
         6: 'NotEqual', 7: 'GEqual', 8: 'Always'}
CULL = {0: 'Off', 1: 'Front', 2: 'Back'}


def get_entry(arr, i):
    item = arr[i]
    return item[0] if isinstance(item, list) else item


def extract_programs(shader, reader_version):
    """Return list of (gpu_program_type, keywords, glsl_text) for GL programs."""
    blob = bytes(shader.compressedBlob)
    out = []
    for i in range(len(shader.platforms)):
        if i >= len(shader.compressedLengths) or i >= len(shader.decompressedLengths):
            break
        clen = get_entry(shader.compressedLengths, i)
        dlen = get_entry(shader.decompressedLengths, i)
        off = get_entry(shader.offsets, i)
        data = CompressionHelper.decompress_lz4(blob[off:off + clen], dlen)
        try:
            prog = ShaderProgram(EndianBinaryReader(data, endian='<'), reader_version)
            for sp in prog.m_SubPrograms:
                if sp.m_ProgramType in GL_PREFERENCE and len(sp.m_ProgramCode) > 100:
                    text = sp.m_ProgramCode.decode('utf8', errors='replace')
                    if '#ifdef VERTEX' in text:
                        out.append((sp.m_ProgramType, tuple(sp.m_Keywords or ()), text))
        except Exception:
            # parser quirk on some blobs: fall back to raw text scan
            try:
                text = data.decode('utf8', errors='replace')
                if '#ifdef VERTEX' in text and '#ifdef FRAGMENT' in text:
                    out.append((ShaderGpuProgramType.kShaderGpuProgramGLCore32, (), text))
            except Exception:
                pass
            continue
    return out


def split_stages(glsl):
    """Split a GLSL program text into {VERTEX, FRAGMENT} sources.

    Stage blocks run from '#ifdef VERTEX' to just before the next stage marker
    (or EOF); nested #ifdef/#endif pairs (e.g. GL_ARB_shader_bit_encoding) are
    kept intact by trimming only the final trailing #endif of the block.
    """
    markers = list(re.finditer(r'#ifdef (VERTEX|FRAGMENT|GEOMETRY)\b', glsl))
    stages = {}
    for i, m in enumerate(markers):
        kind = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(glsl)
        block = glsl[start:end].rstrip()
        if block.endswith('#endif'):
            block = block[:-len('#endif')].rstrip()
        stages[kind] = block
    return stages


def strip_preamble(stage_src):
    """Remove the #version/#extension/#define boilerplate; keep uniforms + code."""
    lines = stage_src.split('\n')
    keep = []
    skip_prefixes = ('#version', '#extension', '#define HLSLCC', '#define UNITY_UNIFORM',
                     '#define UNITY_SUPPORTS_UNIFORM_LOCATION', '#if ', '#else', '#endif', '#ifdef', '#undef')
    for ln in lines:
        s = ln.strip()
        if any(s.startswith(p) for p in skip_prefixes):
            continue
        keep.append(ln)
    return '\n'.join(keep)


def translate_body(glsl_body):
    """Mechanical GLSL -> CG translation of the main() body."""
    body = glsl_body
    # types
    body = re.sub(r'\bvec([234])\b', r'float\1', body)
    body = re.sub(r'\bbvec([234])\b', r'bool\1', body)
    body = re.sub(r'\bivec([234])\b', r'int\1', body)
    body = re.sub(r'\bmat([234])\b', r'float\1x\1', body)
    # functions
    body = re.sub(r'\bmix\s*\(', 'lerp(', body)
    body = re.sub(r'\bfract\s*\(', 'frac(', body)
    body = re.sub(r'\bmod\s*\(', 'fmod(', body)
    body = re.sub(r'\binversesqrt\s*\(', 'rsqrt(', body)
    body = re.sub(r'\bfclamp\s*\(', 'clamp(', body)
    body = re.sub(r'\btextureLod\s*\(\s*(\w+)\s*,\s*([^,]+),\s*0\.0\)', r'tex2Dlod(\1, float4(\2, 0.0, 0.0))', body)
    body = re.sub(r'\btextureLod\s*\(\s*(\w+)\s*,\s*([^,]+),\s*([-\d.]+)\)', r'tex2Dlod(\1, float4(\2, 0.0, \3))', body)
    body = re.sub(r'\btexture\s*\(\s*(\w+)\s*,\s*([^)]+)\)', r'tex2D(\1, \2)', body)
    body = re.sub(r'\batan\s*\(\s*([^,]+),\s*([^)]+)\)', r'atan2(\1, \2)', body)
    body = re.sub(r'\bdFdx\s*\(', 'ddx(', body)
    body = re.sub(r'\bdFdy\s*\(', 'ddy(', body)
    body = re.sub(r'\bgl_FragCoord\b', 'vpos', body)
    return body


def parse_uniforms(stage_src):
    """Extract uniform declarations: name -> ('sampler2D'|'float4'|'float4x4', array_size)."""
    uniforms = {}
    for m in re.finditer(r'uniform\s+(?:vec4|float4)\s+(hlslcc_mtx4x4)?(\w+)(\[\d+\])?\s*;', stage_src):
        is_mtx = bool(m.group(1))
        name = m.group(2)
        uniforms[name] = ('float4x4' if is_mtx else 'float4')
    for m in re.finditer(r'uniform\s+sampler2D\s+(\w+)\s*;', stage_src):
        uniforms[m.group(1)] = 'sampler2D'
    for m in re.finditer(r'uniform\s+(?:float|int)\s+(\w+)\s*;', stage_src):
        uniforms[m.group(1)] = 'float'
    return uniforms


def parse_ios(stage_src, direction):
    """in/out declarations -> list of (type, name)."""
    ios = []
    for m in re.finditer(rf'{direction}\s+(vec[234]|float|vec2)\s+(\w+)\s*;', stage_src):
        t = m.group(1)
        n = m.group(2)
        cg = {'vec4': 'float4', 'vec3': 'float3', 'vec2': 'float2', 'float': 'float'}[t]
        ios.append((cg, n))
    return ios


UNITY_MATRIX_MAP = {
    'unity_ObjectToWorld': 'unity_ObjectToWorld',
    'unity_WorldToObject': 'unity_WorldToObject',
    'unity_MatrixVP': 'unity_MatrixVP',
    'unity_MatrixV': 'unity_MatrixV',
    'unity_MatrixInvV': 'unity_MatrixInvV',
    'unity_WorldToCamera': 'unity_WorldToCamera',
    'unity_CameraToWorld': 'unity_CameraToWorld',
    'unity_MatrixInvVP': 'unity_MatrixInvVP',
    'unity_CameraProjection': 'unity_CameraProjection',
    'unity_CameraInvProjection': 'unity_CameraInvProjection',
}


def translate_matrix_chains(body):
    """Replace HLSLcc matrix multiply chains with mul(vector, matrix).

    Pattern: u_x = v.yyyy * hlslcc_mtx4x4M[1]; u_x = hlslcc_mtx4x4M[0] * v.xxxx + u_x; ...
    This is mul(v, M) in HLSL row-major convention.
    """
    # Find chains: result = vec.comp * mtx[i]; result = mtx[j] * vec.comp + result; ...
    # We detect the final assignment to a target and rewrite the whole chain.
    lines = body.split('\n')
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r'(\s*)(\w+)\s*=\s*(\w+)\.(\w+)\s*\*\s*hlslcc_mtx4x4(\w+)\[(\d+)\]\s*;', ln)
        if m:
            indent, dst, vec, comp, mtx, idx = m.groups()
            # collect the chain
            chain = [(vec, comp, mtx, int(idx))]
            j = i + 1
            final_target = None
            while j < len(lines):
                m2 = re.match(r'(\s*)(\w+)\s*=\s*hlslcc_mtx4x4(\w+)\[(\d+)\]\s*\*\s*(\w+)\.(\w+)\s*\+\s*(\w+)\s*;', lines[j])
                if m2 and m2.group(2) == dst and m2.group(3) == mtx and m2.group(5) == vec and m2.group(7) == dst:
                    chain.append((vec, m2.group(6), mtx, int(m2.group(4))))
                    j += 1
                    continue
                m3 = re.match(r'(\s*)(\w+)\s*=\s*(\w+)\s*\+\s*hlslcc_mtx4x4(\w+)\[(\d+)\]\s*;', lines[j])
                if m3 and m3.group(2) == dst and m3.group(3) == dst and m3.group(4) == mtx:
                    chain.append((vec, 'wwww', mtx, int(m3.group(5))))
                    j += 1
                    continue
                # final: gl_Position = mtx[k] * u.wwww + u2; style
                m4 = re.match(r'(\s*)(gl_Position|\w+)\s*=\s*hlslcc_mtx4x4(\w+)\[(\d+)\]\s*\*\s*(\w+)\.(\w+)\s*\+\s*(\w+)\s*;', lines[j])
                if m4 and m4.group(3) == mtx and m4.group(5) == dst:
                    final_target = m4.group(1) + m4.group(2)
                    chain.append((dst, m4.group(6), mtx, int(m4.group(4))))
                    j += 1
                break
            # sort chain by matrix index and build mul()
            chain.sort(key=lambda c: c[3])
            comps = [c[1] for c in chain]
            if comps == ['xxxx', 'yyyy', 'zzzz', 'wwww'] or comps == ['yyyy', 'xxxx', 'zzzz', 'wwww']:
                target = final_target if final_target else dst
                out.append(f"{indent}{target} = mul({vec}, {mtx});")
                i = j
                continue
            # not a clean 4-term chain; emit as-is with mtx name fixed
            for k in range(i, j):
                out.append(lines[k].replace(f'hlslcc_mtx4x4{mtx}', mtx))
            i = j
            continue
        out.append(ln)
        i += 1
    return '\n'.join(out)


def build_shader(name, prop_block, subshader_state, vertex_cg, fragment_cg, uniforms_vs, uniforms_fs, ios_vs_in, ios_vs_out, ios_fs_in):
    """Assemble the final .shader text."""
    # uniform declarations (skip Unity built-ins provided by UnityCG)
    builtin = set(UNITY_MATRIX_MAP) | {'unity_FogColor', 'unity_FogParams'}
    decls = {}
    for src in (uniforms_vs, uniforms_fs):
        for uname, utype in src.items():
            if uname in builtin:
                continue
            decls[uname] = utype
    decl_text = ''.join(f'\t\t\t{t} {n};\n' for n, t in sorted(decls.items()))

    # v2f fields from vertex outputs
    v2f_fields = []
    assign_in_frag = {}
    sem_map = {}
    texcoord_n = 0
    for t, n in ios_vs_out:
        if n == 'gl_Position':
            continue
        sem = f'TEXCOORD{texcoord_n}'
        texcoord_n += 1
        v2f_fields.append(f'\t\t\t\t{t} {n} : {sem};')
        sem_map[n] = sem

    # appdata fields from vertex inputs
    app_fields = []
    for t, n in ios_vs_in:
        sem = 'POSITION' if 'POSITION' in n else ('COLOR' if 'COLOR' in n else f'TEXCOORD')
        if 'TEXCOORD' in sem:
            idx = re.search(r'(\d+)$', n)
            sem = f'TEXCOORD{idx.group(1) if idx else 0}'
        elif 'COLOR' in n:
            idx = re.search(r'(\d+)$', n)
            sem = f'COLOR{idx.group(1) if idx else 0}'
        app_fields.append(f'\t\t\t\t{t} {n} : {sem};')

    shader = f'''Shader "{name}" {{
{prop_block}
	SubShader
	{{
{subshader_state}
		Pass
		{{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma target 3.0
			#include "UnityCG.cginc"

{decl_text}
			struct appdata_t
			{{
{chr(10).join(app_fields)}
			}};

			struct v2f
			{{
				float4 gl_Position : SV_POSITION;
{chr(10).join(v2f_fields)}
			}};

			v2f vert(appdata_t input)
			{{
				v2f output;
{vertex_cg}
				return output;
			}}

			fixed4 frag(v2f input) : SV_Target
			{{
{fragment_cg}
			}}
			ENDCG
		}}
	}}
}}
'''
    return shader


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shaders = {}
    for f in sorted(glob.glob(f'{GAME_DATA}/*.assets')):
        env = UnityPy.load(f)
        for o in env.objects:
            if o.type.name != 'Shader':
                continue
            s = o.read()
            if not s.m_ParsedForm:
                continue
            name = s.m_ParsedForm.m_Name
            if not (name.startswith('Custom/') or name.startswith('Unlit/')):
                continue
            shaders[name] = (s, o.version)

    print(f"found {len(shaders)} custom shaders")
    for name, (s, ver) in sorted(shaders.items()):
        programs = extract_programs(s, ver)
        if not programs:
            print(f"  SKIP {name}: no GL program")
            continue
        # prefer GLCore32, then fewest keywords (base variant)
        programs.sort(key=lambda p: (GL_PREFERENCE.index(p[0]), len(p[1])))
        ptype, _kw, glsl = programs[0]
        stages = split_stages(glsl)
        if 'VERTEX' not in stages or 'FRAGMENT' not in stages:
            print(f"  SKIP {name}: missing stages ({list(stages)})")
            continue
        vs = strip_preamble(stages['VERTEX'])
        fs = strip_preamble(stages['FRAGMENT'])
        out_name = name.replace('/', '_') + '.glsl'
        (OUT_DIR / out_name).write_text(
            f"// {name}  [{ptype.name}]\n// ===== VERTEX =====\n{vs}\n// ===== FRAGMENT =====\n{fs}\n")
        print(f"  OK {name} <- {ptype.name} ({len(glsl)}b)")
    print("done")


if __name__ == '__main__':
    sys.exit(main())
