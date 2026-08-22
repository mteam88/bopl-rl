#!/usr/bin/env python3
"""Translate extracted HLSLcc GLSL into Unity CGPROGRAM shaders.

Design: keep the translated main() body as close to verbatim as possible.
- GLSL stage locals (u_xlat*, vs_*, in_*, gl_Position, SV_Target0) become
  locals/params with identical names; only semantics-preserving rewrites:
  vecN->floatN, mix->lerp, textureLod->tex2Dlod, matrix mul chains -> mul().
- ShaderLab wrapper (properties/tags/state) comes from the parsed form.
"""
import re
import sys
import glob
import pathlib
import UnityPy
from UnityPy.export.ShaderConverter import ShaderProgram, CompressionHelper, ShaderGpuProgramType
from UnityPy.streams import EndianBinaryReader

GAME_DATA = '/home/mte/Documents/Coding/bopl-re/game/BoplBattle_Data'
SRC_DIR = pathlib.Path('/home/mte/Documents/Coding/bopl-re/original-shaders')
OUT_DIR = pathlib.Path('/home/mte/Documents/Coding/bopl-re/translated-shaders')

GL_PREFERENCE = [
    ShaderGpuProgramType.kShaderGpuProgramGLCore32,
    ShaderGpuProgramType.kShaderGpuProgramGLCore41,
    ShaderGpuProgramType.kShaderGpuProgramGLCore43,
    ShaderGpuProgramType.kShaderGpuProgramGLES31,
    ShaderGpuProgramType.kShaderGpuProgramGLES31AEP,
    ShaderGpuProgramType.kShaderGpuProgramGLES3,
    ShaderGpuProgramType.kShaderGpuProgramGLLegacy,
]

BLEND_FACTORS = {0.0: 'Zero', 1.0: 'One', 2.0: 'DstColor', 3.0: 'SrcColor', 4.0: 'OneMinusDstColor',
                 5.0: 'SrcAlpha', 6.0: 'OneMinusSrcColor', 7.0: 'DstAlpha', 8.0: 'OneMinusDstAlpha',
                 9.0: 'SrcAlphaSaturate', 10.0: 'OneMinusSrcAlpha'}
BLEND_OPS = {0.0: 'Add', 1.0: 'Sub', 2.0: 'RevSub', 3.0: 'Min', 4.0: 'Max'}
ZTEST = {0.0: 'Off', 1.0: 'Never', 2.0: 'Less', 3.0: 'Equal', 4.0: 'LEqual', 5.0: 'Greater',
         6.0: 'NotEqual', 7.0: 'GEqual', 8.0: 'Always'}
CULL = {0.0: 'Off', 1.0: 'Front', 2.0: 'Back'}
COMP = {0.0: 'Disabled', 1.0: 'Never', 2.0: 'Less', 3.0: 'Equal', 4.0: 'LEqual', 5.0: 'Greater',
        6.0: 'NotEqual', 7.0: 'GEqual', 8.0: 'Always'}
STENCIL_OP = {0.0: 'Keep', 1.0: 'Zero', 2.0: 'Replace', 3.0: 'IncrSat', 4.0: 'DecrSat',
              5.0: 'Invert', 6.0: 'IncrWrap', 7.0: 'DecrWrap'}

UNITY_BUILTIN_UNIFORMS = {
    'unity_ObjectToWorld', 'unity_WorldToObject', 'unity_MatrixVP', 'unity_MatrixV',
    'unity_MatrixInvV', 'unity_WorldToCamera', 'unity_CameraToWorld', 'unity_MatrixInvVP',
    'unity_CameraProjection', 'unity_CameraInvProjection', '_ProjectionParams',
    '_ScreenParams', '_Time', '_SinTime', '_CosTime', 'unity_DeltaTime', '_WorldSpaceCameraPos',
    '_ZBufferParams', 'unity_OrthoParams', 'unity_FogColor', 'unity_FogParams',
    'unity_AmbientSky', '_LightColor0', '_WorldSpaceLightPos0',
}


def get_entry(arr, i):
    item = arr[i]
    return item[0] if isinstance(item, list) else item


def extract_programs(shader, reader_version):
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
            try:
                text = data.decode('utf8', errors='replace')
                if '#ifdef VERTEX' in text and '#ifdef FRAGMENT' in text:
                    out.append((ShaderGpuProgramType.kShaderGpuProgramGLCore32, (), text))
            except Exception:
                pass
    return out


def split_stages(glsl):
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
    lines = stage_src.split('\n')
    keep = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(('#version', '#extension', '#define', '#if ', '#ifdef', '#endif', '#else', '#undef')):
            continue
        keep.append(ln)
    return '\n'.join(keep)


def parse_uniforms(src):
    """name -> cg type"""
    uniforms = {}
    for m in re.finditer(r'uniform\s+(?:vec4|float4)\s+(hlslcc_mtx4x4)?(\w+)(\[\d+\])?\s*;', src):
        uniforms[m.group(2)] = 'float4x4' if m.group(1) else 'float4'
    for m in re.finditer(r'uniform\s+sampler2D\s+(\w+)\s*;', src):
        uniforms[m.group(1)] = 'sampler2D'
    for m in re.finditer(r'uniform\s+float\s+(\w+)\s*;', src):
        uniforms[m.group(1)] = 'float'
    for m in re.finditer(r'uniform\s+(?:vec3|vec2|int|bool)\s+(\w+)\s*;', src):
        uniforms[m.group(1)] = 'float4'
    return uniforms


def parse_decls(src, keyword):
    """in/out declarations -> [(cg_type, name)]"""
    out = []
    for m in re.finditer(rf'\b{keyword}\s+(vec4|vec3|vec2|float)\s+(\w+)\s*;', src):
        cg = {'vec4': 'float4', 'vec3': 'float3', 'vec2': 'float2', 'float': 'float'}[m.group(1)]
        out.append((cg, m.group(2)))
    return out


def parse_locals(src):
    """global temp declarations between uniforms and main: vec4 u_xlat0; etc."""
    locals_ = []
    head = src.split('void main()')[0]
    type_map = {'vec4': 'float4', 'vec3': 'float3', 'vec2': 'float2', 'float': 'float',
                'bool': 'bool', 'int': 'int',
                'bvec2': 'bool2', 'bvec3': 'bool3', 'bvec4': 'bool4',
                'ivec2': 'int2', 'ivec3': 'int3', 'ivec4': 'int4'}
    for m in re.finditer(r'^\s*(vec4|vec3|vec2|float|bool|int|bvec2|bvec3|bvec4|ivec2|ivec3|ivec4)\s+(\w+)\s*;\s*$', head, re.M):
        locals_.append((type_map[m.group(1)], m.group(2)))
    return locals_


def get_main_body(src):
    m = re.search(r'void main\(\)\s*\{', src)
    if not m:
        return ''
    start = m.end()
    # find matching closing brace
    depth = 1
    i = start
    while i < len(src) and depth:
        if src[i] == '{':
            depth += 1
        elif src[i] == '}':
            depth -= 1
        i += 1
    return src[start:i - 1]


def split_args(s):
    """Split a function-call argument list at top-level commas."""
    args = []
    depth = 0
    cur = ''
    for ch in s:
        if ch in '(':
            depth += 1
        elif ch in ')':
            depth -= 1
        if ch == ',' and depth == 0:
            args.append(cur)
            cur = ''
        else:
            cur += ch
    args.append(cur)
    return args


def replace_fn_calls(b, names, repl):
    """Paren-aware replacement of fn(a, b) calls. repl(args) -> new text."""
    for name in names:
        while True:
            idx = b.find(name + '(')
            if idx == -1:
                break
            # ensure word boundary
            if idx > 0 and (b[idx - 1].isalnum() or b[idx - 1] == '_'):
                break
            start = idx + len(name) + 1
            depth = 1
            i = start
            while i < len(b) and depth:
                if b[i] == '(':
                    depth += 1
                elif b[i] == ')':
                    depth -= 1
                i += 1
            inner = b[start:i - 1]
            args = split_args(inner)
            new = repl(args)
            b = b[:idx] + new + b[i:]
    return b


CMP_MAP = {'lessThan': '<', 'greaterThan': '>', 'lessThanEqual': '<=',
           'greaterThanEqual': '>=', 'equal': '==', 'notEqual': '!='}


def translate_expr_body(body, type_env=None):
    b = body
    b = re.sub(r'\bvec([234])\s*\(', r'float\1(', b)
    b = re.sub(r'\bvec([234])\b', r'float\1', b)
    b = re.sub(r'\bmat([234])\b', r'float\1x\1', b)
    b = re.sub(r'\bmix\s*\(', 'lerp(', b)
    b = re.sub(r'\bfract\s*\(', 'frac(', b)
    b = re.sub(r'\bmod\s*\(', 'fmod(', b)
    b = re.sub(r'\binversesqrt\s*\(', 'rsqrt(', b)
    b = re.sub(r'\bfclamp\s*\(', 'clamp(', b)
    b = re.sub(r'\btextureLod\s*\(\s*(\w+)\s*,\s*([^,]+),\s*(-?[\d.]+)\s*\)',
               r'tex2Dlod(\1, float4(\2, 0.0, \3))', b)
    b = re.sub(r'\btexture\s*\(\s*(\w+)\s*,\s*([^)]+?)\s*\)', r'tex2D(\1, \2)', b)
    b = re.sub(r'\batan\s*\(', 'atan2(', b)
    b = re.sub(r'\bdFdx\s*\(', 'ddx(', b)
    b = re.sub(r'\bdFdy\s*\(', 'ddy(', b)
    b = re.sub(r'\bbool\s*\(\s*(\w+)\s*\)', r'(\1)', b)
    b = re.sub(r'\bbvec([234])\b', r'bool\1', b)
    b = re.sub(r'\bivec([234])\b', r'int\1', b)
    # GLSL component-wise comparisons -> HLSL comparison operators (yield boolN)
    for fn, op in CMP_MAP.items():
        b = replace_fn_calls(b, [fn], lambda args, op=op: f'(({args[0].strip()}) {op} ({args[1].strip()}))')
    # hlslcc matrix names -> unity builtin names
    b = re.sub(r'hlslcc_mtx4x4(\w+)', r'\1', b)
    # Unity's non-D3D shader frontends reject single-argument vector constructors.
    # Unwrap nested same-type: float2(float2(a, b)) -> float2(a, b)
    for n in (2, 3, 4):
        b = re.sub(rf'\bfloat{n}\s*\(\s*float{n}\s*\(([^()]+)\)\s*\)', rf'float{n}(\1)', b)
    # Type-aware single-argument constructor resolution:
    #   floatN(scalar) -> (scalar).xxx   (splat)
    #   floatN(vectorN) -> vectorN       (copy constructor is identity)
    env = type_env or {}

    def make_resolve(n):
        def resolve_ctor(m):
            expr = m.group(1).strip()
            etype = env.get(expr)
            if etype is None and re.fullmatch(r'-?[\d.]+[fF]?', expr):
                etype = 'float'
            if etype is None and re.search(r'\.[xyzw]+$', expr):
                etype = {1: 'float', 2: 'float2', 3: 'float3', 4: 'float4'}.get(len(expr.rsplit('.', 1)[1]))
            if etype == 'float':
                return f'({expr}).' + ('x' * n)
            if etype == f'float{n}':
                return f'({expr})'
            if etype in ('int2', 'int3', 'int4', 'uint2', 'uint3', 'uint4',
                         'bool2', 'bool3', 'bool4', 'float2', 'float3', 'float4'):
                # vector conversion: expand component-wise (int->float is implicit)
                comps = ['x', 'y', 'z', 'w'][:n]
                return f'float{n}(' + ', '.join(f'{expr}.{c}' for c in comps) + ')'
            # unknown type: emit explicit multi-arg form (safe for scalars)
            return f'float{n}(' + ', '.join([expr] * n) + ')'
        return resolve_ctor

    for n in (2, 3, 4):
        b = re.sub(rf'\bfloat{n}\s*\(\s*([\w.]+)\s*\)(?!\s*\.)', make_resolve(n), b)
    return b


def collapse_matrix_chains(body):
    """Collapse the standard 4-row transform chain into mul(v, M)."""
    lines = body.split('\n')
    out = []
    i = 0
    while i < len(lines):
        m = re.match(r'(\s*)(\w+)\s*=\s*(\w+)\.yyyy\s*\*\s*(\w+)\[1\]\s*;\s*$', lines[i])
        if m and i + 3 < len(lines):
            indent, dst, vec, mtx = m.groups()
            m2 = re.match(rf'\s*{dst}\s*=\s*{mtx}\[0\]\s*\*\s*{vec}\.xxxx\s*\+\s*{dst}\s*;\s*$', lines[i + 1])
            m3 = re.match(rf'\s*{dst}\s*=\s*{mtx}\[2\]\s*\*\s*{vec}\.zzzz\s*\+\s*{dst}\s*;\s*$', lines[i + 2])
            m4a = re.match(rf'\s*{dst}\s*=\s*{dst}\s*\+\s*{mtx}\[3\]\s*;\s*$', lines[i + 3])
            m4b = re.match(rf'\s*{dst}\s*=\s*{mtx}\[3\]\s*\*\s*{vec}\.wwww\s*\+\s*{dst}\s*;\s*$', lines[i + 3])
            if m2 and m3 and (m4a or m4b):
                out.append(f'{indent}{dst} = mul({vec}, {mtx});')
                i += 4
                continue
        # 5-line variant ending in gl_Position
        m = re.match(r'(\s*)(\w+)\s*=\s*(\w+)\.yyyy\s*\*\s*(\w+)\[1\]\s*;\s*$', lines[i])
        if m and i + 3 < len(lines):
            indent, dst, vec, mtx = m.groups()
            m2 = re.match(rf'\s*{dst}\s*=\s*{mtx}\[0\]\s*\*\s*{vec}\.xxxx\s*\+\s*{dst}\s*;\s*$', lines[i + 1])
            m3 = re.match(rf'\s*{dst}\s*=\s*{mtx}\[2\]\s*\*\s*{vec}\.zzzz\s*\+\s*{dst}\s*;\s*$', lines[i + 2])
            m4 = re.match(rf'\s*gl_Position\s*=\s*{mtx}\[3\]\s*\*\s*{vec}\.wwww\s*\+\s*{dst}\s*;\s*$', lines[i + 3])
            if m2 and m3 and m4:
                out.append(f'{indent}gl_Position = mul({vec}, {mtx});')
                i += 4
                continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)


def fmt_default(prop):
    d = (prop.m_DefValue_0_, prop.m_DefValue_1_, prop.m_DefValue_2_, prop.m_DefValue_3_)
    t = prop.m_Type
    if t in (0, 1):  # color / vector
        return f'({d[0]:g},{d[1]:g},{d[2]:g},{d[3]:g})'
    if t in (2, 3):  # float / range
        return f'{d[0]:g}'
    if t == 4:  # texture
        return '"white" {}'
    return '0'


def prop_type_str(prop):
    t = prop.m_Type
    if t == 0:
        return 'Color'
    if t == 1:
        return 'Vector'
    if t == 2:
        return 'Float'
    if t == 3:
        return f'Range({prop.m_DefValue_1_:g}, {prop.m_DefValue_2_:g})'
    if t == 4:
        return '2D'
    return 'Float'


def build_properties(pf):
    lines = ['\tProperties\n\t{']
    for prop in pf.m_PropInfo.m_Props:
        flags = prop.m_Flags
        attr = ''
        if flags & 1 and flags & 2:
            attr = '[HideInInspector] [PerRendererData] '
        elif flags & 1:
            attr = '[HideInInspector] '
        elif flags & 2:
            attr = '[PerRendererData] '
        ptype = prop_type_str(prop)
        default = fmt_default(prop)
        desc = prop.m_Description or prop.m_Name
        lines.append(f'\t\t{attr}{prop.m_Name} ("{desc}", {ptype}) = {default}')
    lines.append('\t}')
    return '\n'.join(lines)


def build_state(state, tags):
    lines = []
    if tags:
        pairs = ' '.join(f'"{k}" = "{v}"' for k, v in tags)
        lines.append(f'\t\tTags {{ {pairs} }}')
    rb = state.rtBlend0
    src = BLEND_FACTORS.get(rb.srcBlend.val, 'One')
    dst = BLEND_FACTORS.get(rb.destBlend.val, 'Zero')
    srcA = BLEND_FACTORS.get(rb.srcBlendAlpha.val, 'One')
    dstA = BLEND_FACTORS.get(rb.destBlendAlpha.val, 'Zero')
    op = BLEND_OPS.get(rb.blendOp.val, 'Add')
    opA = BLEND_OPS.get(rb.blendOpAlpha.val, 'Add')
    if not (rb.srcBlend.val == 1 and rb.destBlend.val == 0 and rb.blendOp.val == 0):
        if (src, dst) != (srcA, dstA) or op != opA:
            lines.append(f'\t\tBlend {src} {dst}, {srcA} {dstA}')
        else:
            lines.append(f'\t\tBlend {src} {dst}')
        if op != 'Add' or opA != 'Add':
            lines.append(f'\t\tBlendOp {op}, {opA}')
    if state.zWrite.val != 1:
        lines.append('\t\tZWrite Off')
    if state.zTest.val != 4:
        lines.append(f'\t\tZTest {ZTEST.get(state.zTest.val, "LEqual")}')
    if state.culling.val != 2:
        lines.append(f'\t\tCull {CULL.get(state.culling.val, "Back")}')
    if state.lighting:
        lines.append('\t\tLighting On')
    return '\n'.join(lines)


def translate_shader(name, pf, vs_src, fs_src):
    vs_uniforms = parse_uniforms(vs_src)
    fs_uniforms = parse_uniforms(fs_src)
    vs_in = parse_decls(vs_src, 'in')
    vs_out = parse_decls(vs_src, 'out')
    fs_in = parse_decls(fs_src, 'in')
    vs_locals = parse_locals(vs_src)
    fs_locals = parse_locals(fs_src)

    # type environment for constructor resolution (uniforms + locals + inputs)
    vs_env = dict(vs_uniforms)
    vs_env.update({n: t for t, n in vs_locals})
    vs_env.update({n: t for t, n in vs_in})
    fs_env = dict(fs_uniforms)
    fs_env.update({n: t for t, n in fs_locals})
    fs_env.update({n: t for t, n in fs_in})
    # matrices collapse to float4x4; treat as non-scalar
    for n in list(vs_env):
        if vs_env[n] == 'float4x4':
            vs_env[n] = 'matrix'
    for n in list(fs_env):
        if fs_env[n] == 'float4x4':
            fs_env[n] = 'matrix'

    vs_body = collapse_matrix_chains(translate_expr_body(get_main_body(vs_src), vs_env))
    fs_body = collapse_matrix_chains(translate_expr_body(get_main_body(fs_src), fs_env))

    # uniform declarations needed (skip Unity built-ins)
    decls = {}
    for src in (vs_uniforms, fs_uniforms):
        for uname, utype in src.items():
            if uname in UNITY_BUILTIN_UNIFORMS:
                continue
            decls[uname] = utype
    decl_text = ''.join(f'\t\t\t{t} {n};\n' for n, t in sorted(decls.items()))

    # v2f fields: vertex outputs (minus gl_Position)
    v2f = []
    texcoord_i = 0
    name_to_field = {}
    for t, n in vs_out:
        if n == 'gl_Position':
            continue
        sem = f'TEXCOORD{texcoord_i}'
        texcoord_i += 1
        v2f.append((t, n, sem))
        name_to_field[n] = n
    v2f_text = '\n'.join(f'\t\t\t\t{t} {n} : {sem};' for t, n, sem in v2f)

    # vertex params from inputs
    def sem_for_input(n):
        if 'POSITION' in n:
            return 'POSITION'
        if 'COLOR' in n:
            idx = re.search(r'(\d+)$', n)
            return f'COLOR{idx.group(1) if idx else 0}'
        m = re.search(r'TEXCOORD(\d+)$', n)
        if m:
            return f'TEXCOORD{m.group(1)}'
        return 'TEXCOORD0'

    params = ', '.join(f'{t} {n} : {sem_for_input(n)}' for t, n in vs_in)

    # strip bare 'return;' from bodies (it would skip output assignment)
    def strip_return(b):
        return '\n'.join(ln for ln in b.split('\n') if ln.strip() != 'return;')
    vs_body = strip_return(vs_body)
    fs_body = strip_return(fs_body)

    # vertex needs locals for every vs_* output + gl_Position
    vs_stage_locals = list(vs_locals)
    vs_stage_locals.append(('float4', 'gl_Position'))
    for t, n, _ in v2f:
        vs_stage_locals.append((t, n))
    vs_local_text = '\n'.join(f'\t\t\t\t{t} {n};' for t, n in vs_stage_locals)
    fs_local_text = '\n'.join(f'\t\t\t\t{t} {n};' for t, n in fs_locals)

    # fragment input locals from v2f
    fs_in_local = '\n'.join(f'\t\t\t\t{t} {n} = input.{n};' for t, n in fs_in if n in name_to_field)

    # vertex: assign outputs at end
    vs_out_assign = '\n'.join(f'\t\t\toutput.{n} = {n};' for _, n, _ in v2f)
    vs_out_assign += '\n\t\t\toutput.gl_Position = gl_Position;'

    prop_block = build_properties(pf)
    state = pf.m_SubShaders[0].m_Passes[0].m_State
    tags = list(pf.m_SubShaders[0].m_Tags.tags) if pf.m_SubShaders[0].m_Tags else []
    state_text = build_state(state, tags)

    shader = f'''Shader "{name}" {{
{prop_block}
	SubShader
	{{
{state_text}
		Pass
		{{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			#pragma target 3.0
			#include "UnityCG.cginc"

{decl_text}
			struct v2f
			{{
				float4 gl_Position : SV_POSITION;
{v2f_text}
			}};

			v2f vert({params})
			{{
				v2f output;
{vs_local_text}
{vs_body}
{vs_out_assign}
				return output;
			}}

			fixed4 frag(v2f input) : SV_Target
			{{
{fs_in_local}
{fs_local_text}
				float4 SV_Target0;
{fs_body}
				return SV_Target0;
			}}
			ENDCG
		}}
	}}
}}
'''
    return shader


PATCHES = {
    # GLSL bool->int bitmask idiom that has no clean HLSL equivalent.
    # Semantics: per-component sign detection of _FlipUV.
    'Custom/FlippableSprite': [
        (r'u_xlati0\.xy = int2\(uvec2\(.*?0xFFFFFFFFu\);',
         'u_xlati0.xy = int2((_FlipUV.x < 0.0) ? -1 : 0, (_FlipUV.y < 0.0) ? -1 : 0);'),
        (r'u_xlati4\.xy = int2\(uvec2\(.*?0xFFFFFFFFu\);',
         'u_xlati4.xy = int2((_FlipUV.x > 0.0) ? -1 : 0, (_FlipUV.y > 0.0) ? -1 : 0);'),
    ],
}


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
        programs.sort(key=lambda p: (GL_PREFERENCE.index(p[0]), len(p[1])))
        ptype, _kw, glsl = programs[0]
        stages = split_stages(glsl)
        if 'VERTEX' not in stages or 'FRAGMENT' not in stages:
            print(f"  SKIP {name}: missing stages")
            continue
        vs = strip_preamble(stages['VERTEX'])
        fs = strip_preamble(stages['FRAGMENT'])
        try:
            shader_text = translate_shader(name, s.m_ParsedForm, vs, fs)
            for pat, rep in PATCHES.get(name, []):
                shader_text = re.sub(pat, rep, shader_text)
            out_name = name.replace('/', '_') + '.shader'
            (OUT_DIR / out_name).write_text(shader_text)
            print(f"  OK {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
    print("done")


if __name__ == '__main__':
    sys.exit(main())
