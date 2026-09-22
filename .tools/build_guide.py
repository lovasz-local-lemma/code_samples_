"""Generate docs/index.html (the reading guide) from the highlights manifest.

Usage: build_guide.py <manifest.json> [repo_base_url]
The manifest is a list of sections, each {section, files:[{path,title,language,why,highlights:[{caption,start,end}],key_functions:[{name,line}]}]}.
Every highlight is read from the file on disk at build time, so the page always shows the current text.
"""
import html, io, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = sys.argv[1]
REPO = (sys.argv[2] if len(sys.argv) > 2 else 'https://github.com/lovasz-local-lemma/code-samples/blob/main').rstrip('/')

TIERS = [
    ('Research', 'Photon primitives and path tracing: the SIGGRAPH branch and the renderer that continues it.',
     ['01-tungsten-photon-primitives', '02-radiancelab']),
    ('Personal projects', 'Systems I built end to end: a Rive rendering studio, a node-based GPU environment, a multiphysics lab and a modal synthesiser, a sweepline Voronoi with the beachline visible.',
     ['03-rivx', '04-shaderweave', '05-simulation', '07-geometry']),
    ('Breadth', 'Smaller pieces that show range: a compact volumetric path tracer with its verification harnesses, a multi-backend FFT with a measured planner, a red-black ordered set with executable invariants, render-graph scheduling, a bilateral grid, a DFA-to-regex construction.',
     ['06-light-transport-compact', '08-numerics-and-engineering']),
]
SECTION_TITLES = {
    '01-tungsten-photon-primitives': 'Photon primitives in Tungsten (SIGGRAPH 2019)',
    '02-radiancelab': 'RadianceLab: hourglass, correlated gather, GPU bidirectional MIS',
    '03-rivx': 'RIVX: Rive rendering and authoring studio',
    '04-shaderweave': 'ShaderWeave: node-based GPU programming environment',
    '05-simulation': 'Crucible2.5D and Modal2D: simulation',
    '06-light-transport-compact': 'Light transport, compact',
    '07-geometry': "Fortune's Loom: sweepline Voronoi and Apollonius diagrams",
    '08-numerics-and-engineering': 'Numerics and engineering',
}
SECTION_NOTES = {
    '01-tungsten-photon-primitives': 'My research branch of Benedikt Bitterli\'s Tungsten. The renderer framework is his; the photon-primitive machinery below is mine, and <a href="../01-tungsten-photon-primitives/CONTRIBUTION.md">CONTRIBUTION.md</a> maps every function with diffs against upstream. The intersection routine here returns a solution object; RadianceLab\'s later solver (section 02) is flat by design — the same mathematics with elegance traded for throughput.',
    '02-radiancelab': 'Excerpts from an 11,000-line integrator and a 4,300-line compute megakernel; each excerpt names its source line range.',
    '03-rivx': 'The geometry core is GL-free and unit-tested; the renderer and 3D-to-vector pieces use the Rive runtime\'s path and matrix types as their interface.',
    '04-shaderweave': 'Hand-written throughout. ImGui, CUDA, NVRTC and OptiX are the interfaces; the runtimes, graph and widgets are mine. The node-editor fork is shown as a diff against upstream.',
    '05-simulation': 'GLSL compute kernels from Crucible2.5D and Eigen-based finite elements and solvers from Modal2D.',
    '06-light-transport-compact': 'A volumetric path tracer small enough to read in one sitting, with the brute-force verification that keeps it honest.',
    '07-geometry': 'The sweep, five beachline balance policies, the numerical kernel under it, a lazy-exact scalar, and Apollonius diagrams with a half-edge builder.',
    '08-numerics-and-engineering': 'Small C++17 modules, each with its tests: signal processing, containers, scheduling, imaging.',
}
HL_LANG = {'cpp': 'cpp', 'glsl': 'glsl', 'cuda': 'cpp', 'crystal': 'crystal', 'ruby': 'ruby', 'diff': 'diff'}

def read_lines(path):
    return io.open(path, encoding='utf-8', errors='replace').read().replace('\r\n', '\n').split('\n')

def dedent(lines):
    ind = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    k = min(ind) if ind else 0
    return [l[k:] if len(l) >= k else l for l in lines]

def esc(s):
    return html.escape(s, quote=True)

manifest = json.load(io.open(MANIFEST, encoding='utf-8'))
by_section = {s['section'].strip('/').split('/')[-1]: s for s in manifest}
missing = []
parts = []
nav = []
file_count = 0
for tier, blurb, keys in TIERS:
    tid = tier.lower().replace(' ', '-')
    parts.append(f'<section class="tier" id="{tid}"><div class="tier-head"><span class="eyebrow">{esc(tier)}</span><p>{esc(blurb)}</p></div>')
    nav.append(f'<li class="nav-tier"><a href="#{tid}">{esc(tier)}</a><ul>')
    for key in keys:
        sec = by_section.get(key)
        if not sec:
            missing.append(key); continue
        parts.append(f'<div class="section" id="{key}"><h2>{esc(SECTION_TITLES.get(key, key))}</h2><p class="section-note">{SECTION_NOTES.get(key, "")}</p>')
        nav.append(f'<li><a href="#{key}">{esc(SECTION_TITLES.get(key, key).split(":")[0])}</a></li>')
        for f in sec['files']:
            rel = f'{key}/{f["path"]}'
            full = os.path.join(ROOT, rel)
            if not os.path.exists(full):
                missing.append(rel); continue
            lines = read_lines(full)
            file_count += 1
            fid = rel.replace('/', '-').replace('.', '-')
            gh = f'{REPO}/{rel}'
            parts.append(f'<article class="file" id="{fid}"><header><h3>{esc(f["title"])}</h3><div class="file-meta"><code>{esc(rel)}</code><span>{len(lines):,} lines</span><a href="{esc(gh)}" target="_blank" rel="noopener">open on GitHub ↗</a><a href="../{esc(rel)}">raw file</a></div></header>')
            parts.append(f'<p class="why">{esc(f["why"])}</p>')
            if f.get('key_functions'):
                kf = ' '.join(f'<a href="{esc(gh)}#L{k["line"]}" target="_blank" rel="noopener"><code>{esc(k["name"])}</code><small>L{k["line"]}</small></a>' for k in f['key_functions'])
                parts.append(f'<div class="keyfns">{kf}</div>')
            for h in f['highlights']:
                a, b = max(1, int(h['start'])), min(len(lines), int(h['end']))
                if b < a: continue
                code = '\n'.join(dedent(lines[a - 1:b]))
                lang = HL_LANG.get(f.get('language', 'cpp'), 'cpp')
                parts.append(f'<figure class="hl"><figcaption><span>{esc(h["caption"])}</span><a href="{esc(gh)}#L{a}-L{b}" target="_blank" rel="noopener">lines {a}–{b} ↗</a></figcaption><pre><code class="language-{lang}">{esc(code)}</code></pre></figure>')
            parts.append('</article>')
        parts.append('</div>')
    nav.append('</ul></li>')
    parts.append('</section>')

body = '\n'.join(parts)
navhtml = '\n'.join(nav)
page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Code samples — Shaojie Jiao</title>
<meta name="description" content="A reading guide to selected source from Shaojie Jiao's rendering research, graphics tools, simulations and mathematics projects.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark-dimmed.min.css">
<style>
:root{{color-scheme:dark;--gold:#e5bf57;--gold-soft:#be9e50;--bg:#0c0d0c;--surface:#141613;--surface2:#1a1c18;--ink:#f0eee5;--muted:#b5b7ad;--line:rgba(221,209,167,.17);--mint:#a6d7bc;--serif:'Playfair Display',Georgia,serif;--sans:Inter,system-ui,sans-serif;--mono:'JetBrains Mono',Consolas,monospace}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;scroll-padding-top:24px}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--sans)}}
a{{color:var(--gold);text-decoration:none}}a:hover{{text-decoration:underline}}
code,pre{{font-family:var(--mono)}}
.layout{{display:grid;grid-template-columns:260px minmax(0,1fr);gap:40px;max-width:1440px;margin:0 auto;padding:0 32px}}
nav.side{{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:36px 0 40px;border-right:1px solid var(--line);font-size:14px}}
nav.side .brand{{font-family:var(--serif);font-size:22px;color:var(--ink);display:block;margin-bottom:4px}}
nav.side .brand small{{display:block;font-family:var(--sans);font-size:12px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-top:6px}}
nav.side ul{{list-style:none;padding:0;margin:0}}nav.side>ul{{margin-top:26px}}
nav.side .nav-tier>a{{display:block;color:var(--gold);letter-spacing:.1em;text-transform:uppercase;font-size:11px;margin:18px 0 6px}}
nav.side .nav-tier li a{{display:block;color:var(--muted);padding:4px 0 4px 10px;border-left:1px solid var(--line)}}nav.side .nav-tier li a:hover,nav.side .nav-tier li a.current{{color:var(--ink);border-left-color:var(--gold);text-decoration:none}}
nav.side .links{{margin-top:28px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}nav.side .links a{{display:block;margin:6px 0}}
main{{padding:36px 0 80px;min-width:0}}
.hero h1{{font-family:var(--serif);font-weight:500;font-size:clamp(34px,4vw,50px);letter-spacing:-.02em;margin:8px 0 14px;line-height:1.15}}
.hero .eyebrow{{color:var(--gold);font-size:12px;letter-spacing:.12em;text-transform:uppercase}}
.hero p{{max-width:820px;color:var(--muted);font-size:17px}}
.hero .note{{border:1px solid var(--line);background:var(--surface);border-radius:8px;padding:18px 22px;color:var(--ink);font-size:16px;max-width:900px;margin:22px 0}}
.hero .note strong{{color:var(--gold);font-weight:600}}
.start{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:26px 0 8px}}
.start a{{display:block;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;color:var(--ink);transition:border-color .2s}}
.start a:hover{{border-color:var(--gold-soft);text-decoration:none}}.start a small{{display:block;color:var(--gold);font-size:11px;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px}}.start a span{{display:block;color:var(--muted);font-size:13px;margin-top:6px}}
.tier{{margin-top:64px;border-top:1px solid var(--line);padding-top:26px}}
.tier-head .eyebrow{{color:var(--gold);font-size:12px;letter-spacing:.12em;text-transform:uppercase}}.tier-head p{{max-width:820px;color:var(--muted);margin:8px 0 0}}
.section{{margin-top:44px}}.section h2{{font-family:var(--serif);font-weight:500;font-size:30px;letter-spacing:-.015em;margin:0 0 6px}}
.section-note{{color:var(--muted);max-width:860px;margin:0 0 22px;font-size:15px}}
article.file{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin:18px 0}}
article.file h3{{margin:0 0 6px;font-size:20px;font-weight:600}}
.file-meta{{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;font-size:13px;color:var(--muted)}}.file-meta code{{color:var(--mint);font-size:12.5px}}
.why{{margin:14px 0 10px;max-width:900px}}
.keyfns{{display:flex;flex-wrap:wrap;gap:6px 8px;margin:6px 0 14px}}.keyfns a{{background:var(--surface2);border:1px solid var(--line);border-radius:6px;padding:3px 9px;font-size:12.5px;color:var(--ink)}}.keyfns a small{{color:var(--muted);margin-left:6px}}.keyfns a:hover{{border-color:var(--gold-soft);text-decoration:none}}
figure.hl{{margin:16px 0 0}}figure.hl figcaption{{display:flex;justify-content:space-between;gap:16px;align-items:baseline;font-size:14px;color:var(--ink);margin-bottom:6px}}figure.hl figcaption a{{font-size:12px;white-space:nowrap;color:var(--muted)}}
pre{{margin:0;border:1px solid var(--line);border-radius:8px;overflow:auto;max-height:560px;font-size:12.5px;line-height:1.5}}pre code.hljs{{padding:14px 16px;background:#0f110f}}
footer{{margin-top:70px;padding-top:24px;border-top:1px solid var(--line);color:var(--muted);font-size:14px;max-width:900px}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr;padding:0 16px}}nav.side{{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line);padding:20px 0}}nav.side>ul{{columns:2;margin-top:10px}}}}
</style>
</head>
<body>
<div class="layout">
<nav class="side" aria-label="Contents">
<a class="brand" href="#top">Shaojie Jiao<small>Code samples · reading guide</small></a>
<ul>
{navhtml}
</ul>
<div class="links"><a href="https://lovasz-local-lemma.github.io/">Portfolio ↗</a><a href="{esc(REPO)}" target="_blank" rel="noopener">Repository ↗</a><a href="mailto:shaojie.jiao.gr@dartmouth.edu">shaojie.jiao.gr@dartmouth.edu</a></div>
</nav>
<main id="top">
<section class="hero">
<span class="eyebrow">Rendering · algorithms · GPU systems</span>
<h1>Code samples, with the good parts marked</h1>
<p>Selected source from my rendering research, graphics tools, simulations and geometry projects. Each file below is introduced in a sentence, its entry points are linked, and the passages worth a reviewer's minute are shown inline. The systems themselves, with recordings and live demos, are on the <a href="https://lovasz-local-lemma.github.io/">portfolio</a>.</p>
<div class="note"><strong>What you are looking at.</strong> To make inspection easy I have lifted these files out of their projects rather than pointing at whole repositories. Everything shown is my own work; external code appears only where it is the interface my code plugs into (Tungsten's renderer framework, the Rive runtime's path types, ImGui, CUDA/OptiX), and it is named as such. For the Tungsten research branch the boundary is made explicit with diffs against upstream. Files were cleaned of commented-out code and working notes before publication, comments only: a script in the repository checks that the code token stream is unchanged.</div>
<div class="start">
<a href="#02-radiancelab-pp_hourglass_sheet_solver-cpp"><small>Start here · research</small>The photon hourglass<span>Refraction chain, double-precision sheet solver, the Jacobian from the discriminant.</span></a>
<a href="#01-tungsten-photon-primitives"><small>Research</small>Photon primitives in Tungsten<span>Analytic ray–primitive intersection and closed-form Jacobians, mapped against upstream.</span></a>
<a href="#03-rivx-render-gl_shape_renderer-cpp"><small>Personal project</small>Stencil-and-cover rasterizer<span>Wrap-mode clip masks, jump-flood feathering, content-hashed tile cache.</span></a>
<a href="#05-simulation-crucible-mpm_p2g-comp"><small>Personal project</small>MLS-MPM particle-to-grid<span>Snow, sand, anisotropic damage and phase-field fracture in one kernel.</span></a>
<a href="#04-shaderweave-CodeNodeRuntime-cpp"><small>Personal project</small>CUDA-JIT and OptiX runtimes<span>NVRTC to PTX behind a PImpl boundary, every failure path unwound.</span></a>
</div>
</section>
{body}
<footer>
<p><strong>Provenance.</strong> Every file here is code I wrote or directed. Third-party code is not redistributed except as diffs against it (Tungsten, <code>imgui-node-editor</code>), and every notice is kept.</p>
<p>My code is MIT-licensed; the Tungsten excerpts keep Tungsten's license, and <code>imgui-node-editor</code> keeps its MIT notice. {file_count} files.</p>
</footer>
</main>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/glsl.min.js"></script>
<script>
hljs.highlightAll();
// In-page links: scroll directly, so navigation works even where fragment navigation is disabled
// (sandboxed previews, data: URLs). The hash is updated when the viewer allows it.
document.addEventListener('click', function (e) {{
  var a = e.target.closest('a[href^="#"]'); if (!a) return;
  var id = decodeURIComponent(a.getAttribute('href').slice(1)); var el = id ? document.getElementById(id) : document.body;
  if (!el) return; e.preventDefault(); el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  try {{ history.replaceState(null, '', '#' + id); }} catch (_) {{}}
}});
// Current-section marker in the contents rail.
(function () {{
  var links = Array.prototype.slice.call(document.querySelectorAll('nav.side .nav-tier li a'));
  var targets = links.map(function (a) {{ return document.getElementById(a.getAttribute('href').slice(1)); }}).filter(Boolean);
  if (!('IntersectionObserver' in window) || !targets.length) return;
  var current = null;
  var io = new IntersectionObserver(function (entries) {{
    entries.forEach(function (en) {{ if (en.isIntersecting) current = en.target.id; }});
    links.forEach(function (a) {{ a.classList.toggle('current', a.getAttribute('href') === '#' + current); }});
  }}, {{ rootMargin: '-10% 0px -80% 0px' }});
  targets.forEach(function (t) {{ io.observe(t); }});
}})();
</script>
</body>
</html>
'''
out = os.path.join(ROOT, 'docs', 'index.html')
os.makedirs(os.path.dirname(out), exist_ok=True)
io.open(out, 'w', encoding='utf-8', newline='\n').write(page)
print(f'wrote {out}: {file_count} files, {len(page):,} bytes')
if missing:
    print('MISSING:', missing)
