"""
Smoke tests for the Streamlit dashboard.

The dashboard is the deliverable most likely to break silently: it is not
imported by anything else, its errors surface as a red box in a browser rather
than a failing build, and it touches every model in the project. Streamlit's
own test harness executes the page headlessly, so it can be checked like any
other code.

These are marked slow because a full run loads the F3 volume and executes the
denoiser; they skip themselves when the data or checkpoints are absent so the
suite still passes on a fresh clone.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
F3 = Path("data/raw/f3/data/train/train_seismic.npy")
LEGACY_CKPT = Path("runs/default/denoiser.pt")

needs_data = pytest.mark.skipif(
    not (F3.exists() and LEGACY_CKPT.exists()),
    reason="F3 volume or checkpoints not present; run data/download_f3.py and training first",
)


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.run()
    return at


@needs_data
def test_dashboard_runs_without_raising(app):
    """The whole page executes top to bottom with no uncaught exception."""
    assert not app.exception, "\n".join(str(getattr(e, "value", e)) for e in app.exception)


@needs_data
def test_dashboard_reports_no_errors_to_the_user(app):
    assert len(app.get("error")) == 0, [str(e.value) for e in app.get("error")]


@needs_data
def test_every_tab_rendered_its_heading(app):
    """Six tabs, each of which must actually produce content rather than
    failing quietly inside a `with tab:` block."""
    headings = " | ".join(str(s.value) for s in app.get("subheader"))
    for expected in ["input, denoised, and what was removed",
                     "Fault segmentation",
                     "scrub through real inlines",
                     "Facies segmentation",
                     "Export",
                     "Diagnostics"]:
        assert expected in headings, f"missing tab content: {expected!r}"


@needs_data
def test_denoise_tab_shows_the_honest_metric_set(app):
    """Leakage must never appear without energy-removed beside it: a filter
    that removes nothing scores a perfect leakage, so the pair is what makes
    the number readable."""
    labels = [m.label for m in app.get("metric")]
    assert "Signal leakage" in labels
    assert "Energy removed" in labels, "leakage shown without the context that makes it honest"


@needs_data
def test_dashboard_uses_no_removed_streamlit_apis():
    """`use_container_width` was removed after 2025-12-31."""
    source = APP.read_text(encoding="utf-8")
    assert "use_container_width" not in source, "deprecated Streamlit API still in use"


@needs_data
def test_facies_tab_labels_in_sample_numbers_as_in_sample(app):
    """Inline 0 is inside the training block. Reporting its accuracy without
    saying so is what made the old 94.4% figure misleading."""
    source = APP.read_text(encoding="utf-8")
    assert "inside* the training block" in source or "inside the training block" in source, \
        "the in-sample section is not labelled as such"
    helps = " ".join(str(m.help or "") for m in app.get("metric"))
    assert "goodness of fit" in helps.lower() or "in-sample" in helps.lower()


@needs_data
@pytest.mark.skipif(not Path("runs/denoise/blindspot.pt").exists(),
                    reason="benchmark denoiser not trained")
def test_controlled_demo_reaches_the_benchmark_result(app):
    """Guards a normalisation bug that is silent by construction.

    The demo adds known noise to a held-out inline and denoises it, so it must
    reproduce roughly what `deepseis.evaluate` reports (~14 dB). Feeding
    `serve_section` pre-normalised data makes it normalise twice; because the
    network is scale equivariant while sigma_n is a fixed physical level, the
    result is not an obvious blow-up but quiet under-denoising -- it cost
    3.17 dB and a 4x drop in energy removed when it was introduced here, with
    the page still rendering perfectly.
    """
    labels = {m.label: m.value for m in app.get("metric")}
    assert "SNR after" in labels, "the controlled demo did not render"

    snr = float(str(labels["SNR after"]).split()[0])
    assert snr > 12.0, (
        f"controlled demo reached only {snr:.2f} dB where the benchmark reports ~14 dB; "
        "check that serve_section is being handed raw-amplitude data"
    )


@needs_data
@pytest.mark.skipif(not Path("runs/denoise/evaluation_final.json").exists(),
                    reason="evaluation report not generated")
def test_benchmark_table_is_read_from_the_evaluation_report(app):
    """The app must not carry its own hand-typed copy of the numbers."""
    frames = app.get("dataframe")
    assert frames, "no benchmark table rendered"
    joined = " ".join(str(f.value) for f in frames)
    assert "Identity (removes nothing)" in joined, "the identity control is missing"
    assert "f-x deconvolution" in joined, "the strongest classical baseline is missing"


@needs_data
def test_export_produces_a_downloadable_file(app):
    """The export flow is two-step (generate, then download), so the download
    button only exists after the click -- which means a broken SEG-Y writer
    would never show up on a plain page render."""
    buttons = [b for b in app.get("button") if "Generate export" in str(b.label)]
    assert buttons, "no export button on the page"

    after = buttons[0].click().run(timeout=600)
    assert not after.exception, "\n".join(
        str(getattr(e, "value", e)) for e in after.exception)
    assert len(after.get("download_button")) >= 1, "clicking export produced no download"
    assert any("Export ready" in str(s.value) for s in after.get("success"))


@needs_data
@pytest.mark.skipif(not Path("runs/default/facies_metrics.json").exists(),
                    reason="facies head not fitted")
def test_units_scored_counts_only_units_present_in_held_out_truth(app):
    """A unit absent from the held-out block but predicted anyway has IoU 0.0,
    not NaN, so counting non-NaN entries claimed 6/6 where only 4 units can be
    scored -- overstating the coverage of the headline number."""
    import json

    meta = json.loads(Path("runs/default/facies_metrics.json").read_text())
    expected = len(meta["scored_classes"])

    values = {m.label: str(m.value) for m in app.get("metric")}
    assert "Units scored" in values, "the held-out summary did not render"
    assert values["Units scored"].startswith(f"{expected} /"), (
        f"reported {values['Units scored']} but only {expected} units occur in the held-out block"
    )


@needs_data
def test_cached_loaders_invalidate_when_their_artifacts_change():
    """Streamlit caches key on arguments, so a loader taking none can never
    invalidate -- retraining a model would leave the page serving the previous
    one indefinitely, with no error and no visible sign. Every loader that
    reads an artifact must take a key derived from that artifact's mtime.
    """
    import ast

    tree = ast.parse(APP.read_text(encoding="utf-8"))
    cached = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            decorated = any(
                "cache_data" in ast.unparse(d) or "cache_resource" in ast.unparse(d)
                for d in node.decorator_list)
            if decorated:
                cached[node.name] = [a.arg for a in node.args.args]

    # loaders that read a trained artifact off disk
    artifact_loaders = ["load_models", "load_blindspot", "load_benchmark_denoiser",
                        "load_facies_metrics", "load_benchmark_table", "make_controlled_demo"]
    for name in artifact_loaders:
        assert name in cached, f"{name} is not a cached loader any more; update this test"
        assert "cache_key" in cached[name], (
            f"{name} is cached but has no mtime-derived cache_key, so it will keep "
            f"serving a stale artifact after retraining"
        )

    source = APP.read_text(encoding="utf-8")
    for name in artifact_loaders:
        assert f"{name}(" in source
    assert source.count("artifact_key(") >= len(artifact_loaders), \
        "some cached loader is called without an artifact_key(...) argument"


def test_dashboard_never_reads_the_whole_survey_into_memory():
    """F3 is stored as float64: a full read is 573 MB before any float32 copy,
    and the app used to do it twice. That alone exhausts a 1 GB hosting tier,
    which is the failure the deployed app has already hit once. Every survey
    read must be memory-mapped and sliced per section."""
    source = APP.read_text(encoding="utf-8")

    assert "mmap=True" in source, "the survey is no longer memory-mapped"
    # the full-survey handle must be cache_resource: cache_data pickles what it
    # stores, which would materialise the memory map straight back into RAM
    idx = source.index("def load_f3_full")
    decorator = source[max(0, idx - 200):idx]
    assert "cache_resource" in decorator, \
        "load_f3_full is cached with cache_data, which defeats the memory map"


def test_data_is_loaded_from_the_hugging_face_dataset_when_absent_locally():
    """The deployed app has no local data, so the HF path is what production
    actually runs."""
    source = APP.read_text(encoding="utf-8")
    assert "hf_hub_download" in source
    assert "Seismic_Data" in source or "DEEPSEIS_HF_DATASET" in source
    # and a local copy must still take precedence, so a dev machine does not
    # re-download hundreds of MB on every run
    assert "local_candidates" in source and "Path(candidate).exists()" in source,         "the resolver no longer prefers a local copy over the download"


@needs_data
def test_3d_tab_renders_a_scene(app):
    headings = " | ".join(str(s.value) for s in app.get("subheader"))
    assert "3D view" in headings, "the 3D tab produced no content"


@needs_data
def test_3d_scene_is_wired_to_the_geometry_module(app):
    """The scene must be built from `viz3d`, whose placement is tested directly
    in tests/test_viz3d.py, and must carry both vertical planes plus horizons.

    Asserted against the source rather than the rendered figure: AppTest cannot
    read `.value` off a plotly_chart that was given an explicit key without
    tripping a session-state lookup.
    """
    source = APP.read_text(encoding="utf-8")
    assert "viz3d.inline_plane" in source and "viz3d.crossline_plane" in source,         "the 3D scene no longer builds its planes from the tested geometry module"
    assert "go.Surface(" in source, "no surface trace in the scene"
    assert "viz3d.horizon_polylines" in source, "horizons are not drawn"
    assert "viz3d.symmetric_limits" in source,         "colour limits are not symmetric, so seismic polarity will read wrong"
    assert "aspectmode=\"manual\"" in source, "the cube is not given an explicit aspect ratio"


@needs_data
def test_3d_tab_states_that_the_time_slice_is_not_denoised(app):
    """A time slice has a different geometry from anything the denoiser was
    trained on. Showing it silently alongside denoised planes would imply it
    had been denoised too."""
    warnings = " ".join(str(w.value) for w in app.get("warning")).lower()
    assert "raw time slices" in warnings or "time slice is raw" in warnings, \
        f"the tab does not say the time slice is undenoised: {warnings[:200]}"


@needs_data
def test_3d_tab_offers_cuboid_and_chair_modes_with_separation(app):
    """The block display and the pull-apart are the point of this tab."""
    source = APP.read_text(encoding="utf-8")
    for token, why in [
        ("viz3d.shell_offsets", "the cuboid shell is not assembled"),
        ("viz3d.apply_notch", "the chair cut is missing"),
        ("viz3d.offset_plane", "separation does not move the faces"),
    ]:
        assert token in source, why

    radios = [str(r.label) for r in app.get("radio")]
    assert any("Display" in r for r in radios), "no display-mode selector"
    sliders = [str(s.label) for s in app.get("slider")]
    assert any("Separation" in s for s in sliders), "no separation slider"


@needs_data
def test_separation_is_a_translation_not_a_recompute(app):
    """Moving the separation slider must not trigger denoising again -- the
    shell is fixed at the survey edges and cached, so only geometry changes."""
    source = APP.read_text(encoding="utf-8")
    sep_block = source[source.index('key="d3_sep"'):source.index("st.plotly_chart(fig3d")]
    assert "offset_plane" in sep_block
    assert "denoised_inline(separation" not in sep_block


@needs_data
def test_3d_tab_detects_a_stale_viz3d_module(app):
    """Streamlit keeps imported modules in sys.modules across reruns, so a
    deploy can leave a new dashboard calling an old `viz3d`. On Streamlit Cloud
    the resulting AttributeError has its message redacted, so the cause is
    invisible unless the app names it."""
    source = APP.read_text(encoding="utf-8")
    assert "_VIZ3D_REQUIRED" in source, "no staleness guard on the 3D tab"
    assert "Reboot app" in source, "the guard does not tell the user how to fix it"

    # every helper the tab calls must be listed in the guard, or the guard
    # gives false confidence
    import re
    called = set(re.findall(r"viz3d\.(\w+)\(", source))
    guard = set(re.findall(r'"(\w+)"', source[source.index("_VIZ3D_REQUIRED"):
                                              source.index("with tab_3d:")]))
    assert called <= guard, f"viz3d helpers used but not guarded: {sorted(called - guard)}"


def test_viz3d_exposes_everything_the_dashboard_guards_for():
    """The guard list must not drift away from the module it guards."""
    import re

    from deepseis import viz3d as _v

    source = APP.read_text(encoding="utf-8")
    guard = re.findall(r'"(\w+)"', source[source.index("_VIZ3D_REQUIRED"):
                                          source.index("with tab_3d:")])
    for name in guard:
        assert hasattr(_v, name), f"guard names {name!r}, which viz3d does not define"
