import os
import io
import glob
import base64
import tempfile
import subprocess
from collections import Counter
import time

import cv2
import numpy as np
import imageio_ffmpeg
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

CONF = 0.25
IOU = 0.50
MAX_SECONDS = 10
MAX_WIDTH = 480
EXAMPLES = sorted(
    glob.glob("examples/*.jpg") + glob.glob("examples/*.jpeg") + glob.glob("examples/*.png")
)[:3]
VIDEO_EXAMPLES = sorted(
    glob.glob("examples/*.mp4") + glob.glob("examples/*.mov")
    + glob.glob("examples/*.avi") + glob.glob("examples/*.mkv")
)[:1]

# Colab me best_model.val() chala ke ye numbers bhar do
METRICS = {"mAP50": "92.5%", "Precision": "85.5%", "Recall": "88.3%", "Epochs": "40"}
GREEN = (94, 197, 34)
RED = (68, 68, 239)

st.set_page_config(page_title="Helmet Detection", page_icon="🪖", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    [data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer, header {display: none;}
    [data-testid="stHeaderActionElements"] {display: none !important;}

    /* Page container: centered, comfortable max width, responsive side padding */
    .block-container {
        max-width: 1180px !important;
        margin: 0 auto !important;
        padding: 1.5rem clamp(1rem, 4vw, 3rem) 2.5rem clamp(1rem, 4vw, 3rem) !important;
    }

    /* Even vertical rhythm between all Streamlit elements */
    [data-testid="stVerticalBlock"] {gap: 1rem !important;}
    [data-testid="stMarkdownContainer"] h4 {
        margin: 1rem 0 .9rem 0 !important; padding: 0 !important;
    }
    [data-testid="stHorizontalBlock"] {margin-top: .5rem;}
    [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"] {gap: 1rem !important;}

    .stApp {
        background:
          radial-gradient(700px 380px at 12% -5%, rgba(20,184,166,.12), transparent),
          radial-gradient(600px 360px at 95% 15%, rgba(14,165,233,.09), transparent),
          #070d14;
    }

    /* Detected image */
    .center-img {text-align: center; width: 100%; margin: 1.5rem 0 2rem 0;}
    .center-img img {
        display: block; margin: 0 auto;
        max-height: 65vh; max-width: 100%; width: auto; height: auto;
        object-fit: contain; border-radius: 14px;
    }
    [data-testid="stImage"] img {border-radius: 12px;}
    video {
        display: block; margin: 0 auto;
        max-height: 65vh; width: 100%; object-fit: contain;
        background: #000; border-radius: 14px;
    }

    /* Hero */
    .hero {
        position: relative; overflow: hidden; text-align: center;
        background: linear-gradient(135deg, #115e59 0%, #155e75 55%, #1e40af 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 24px; padding: 2.6rem 1.5rem 2.2rem 1.5rem; margin: 0 0 .5rem 0;
        box-shadow: 0 16px 40px rgba(0,0,0,.4);
    }
    .hero::before {
        content: ""; position: absolute; width: 300px; height: 300px; right: -90px; top: -130px;
        background: radial-gradient(circle, rgba(255,255,255,.07), transparent 65%);
    }
    .hero > * {position: relative; z-index: 1;}
    .badge {
        display: inline-block; font-size: .72rem; letter-spacing: .14em; font-weight: 600;
        color: #d1fae5; background: rgba(0,0,0,.2); border: 1px solid rgba(255,255,255,.2);
        border-radius: 999px; padding: .3rem .95rem;
    }
    .hero h1 {
        color: #f1f5f9; font-size: 2.6rem; font-weight: 700;
        margin: 1rem 0 .5rem 0; padding: 0; line-height: 1.15;
    }
    .hero p {color: rgba(226,232,240,.8); margin: 0 0 1.3rem 0; font-size: 1.04rem;}
    .chip {
        display: inline-block; color: #e2e8f0; background: rgba(0,0,0,.22);
        border: 1px solid rgba(255,255,255,.15);
        padding: .28rem .85rem; border-radius: 999px; margin: .2rem; font-size: .8rem; font-weight: 500;
    }

    /* Info strips */
    .strip {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 0 0 .8rem 0;}
    .strip4 {display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 0 0 1rem 0;}
    .info {
        background: #0e1823; border: 1px solid #172a38; border-radius: 14px;
        padding: .9rem 1rem; text-align: center;
    }
    .info .k {color: #5b8196; font-size: .7rem; letter-spacing: .12em; font-weight: 600;}
    .info .v {color: #cbd5e1; font-size: 1rem; font-weight: 600; margin-top: .2rem;}

    /* Tabs */
    [data-baseweb="tab-list"] {gap: 28px !important; border-bottom: 1px solid #172a38;}
    [data-baseweb="tab-panel"] {padding-top: 1.4rem !important;}
    button[role="tab"] {
        background: transparent !important;
        border: none !important;
        padding: .6rem .1rem !important;
        height: auto !important;
        font-weight: 600; color: #7b8da0 !important;
    }
    button[role="tab"][aria-selected="true"] {
        background: transparent !important;
        color: #5eead4 !important;
    }
    [data-baseweb="tab-highlight"] {background-color: #14b8a6 !important; height: 2px !important;}
    [data-baseweb="tab-border"] {background-color: #172a38 !important;}

    /* Uploader */
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed rgba(45,212,191,.4); border-radius: 18px;
        background: #0e1823; padding: 1.6rem; transition: all .2s;
    }
    [data-testid="stFileUploader"] section:hover {border-color: #2dd4bf;}

    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderPagination"],
    [data-testid="stFileUploaderDeleteBtn"] {display: none !important;}
    [data-testid="stFileUploaderDropzoneInstructions"] {display: none !important;}

    /* Steps */
    .steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: .4rem 0 1.6rem 0;}
    .step {
        background: #0e1823; border: 1px solid #172a38; border-radius: 18px;
        padding: 1.4rem 1.2rem; transition: all .25s ease;
    }
    .step:hover {
        border-color: #14b8a6 !important;
        background: #10202c !important;
        transform: translateY(-5px);
        box-shadow: 0 16px 36px rgba(20,184,166,.20);
    }
    .ic {
        width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center;
        justify-content: center; font-size: 1.3rem; margin-bottom: .9rem; transition: transform .25s ease;
    }
    .step:hover .ic {transform: scale(1.1);}
    .i1 {background: #115e59;} .i2 {background: #155e75;} .i3 {background: #1e3a8a;}
    .step h4 {margin: 0 0 .4rem 0 !important; color: #f1f5f9; font-size: 1.05rem;}
    .step p {color: #94a3b8; font-size: .9rem; margin: 0; line-height: 1.55;}

    /* Results */
    .stat {border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1.2rem;}
    .stat .n {font-size: 2.6rem; font-weight: 700; line-height: 1;}
    .stat .l {color: #cbd5e1; font-size: .92rem; margin-top: .4rem;}
    .good {background: rgba(34,197,94,.09); border: 1px solid rgba(34,197,94,.35);}
    .good .n {color: #4ade80;}
    .bad  {background: rgba(239,68,68,.09); border: 1px solid rgba(239,68,68,.35);}
    .bad .n  {color: #f87171;}

    .banner {border-radius: 12px; padding: .9rem 1.2rem; margin: 0 0 1rem 0; font-weight: 600;}
    .ok   {background: rgba(34,197,94,.1); border: 1px solid #166534; color: #86efac;}
    .warn {background: rgba(239,68,68,.1); border: 1px solid #991b1b; color: #fca5a5;}

    /* About */
    .about {display: grid; grid-template-columns: 1.6fr 1fr; gap: 16px; margin: 0 0 1rem 0;}
    .card {
        background: #0e1823; border: 1px solid #172a38; border-radius: 18px; padding: 1.4rem;
    }
    .card .k {color: #5b8196; font-size: .7rem; letter-spacing: .12em; font-weight: 600; margin-bottom: .7rem;}
    .card p {color: #94a3b8; font-size: .92rem; line-height: 1.65; margin: 0;}
    .tag {
        display: inline-block; color: #cbd5e1; background: #0a1520; border: 1px solid #1f3547;
        padding: .25rem .75rem; border-radius: 8px; margin: 0 .35rem .45rem 0; font-size: .8rem;
    }
    .legend {color: #94a3b8; font-size: .88rem; margin-top: .7rem;}
    .dot {display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: .45rem;}

    /* Footer */
    .foot2 {
        text-align: center; color: #5b8196; font-size: .85rem;
        border-top: 1px solid #172a38; margin-top: 2.5rem; padding-top: 1.4rem;
    }
    .foot2 a {color: #5eead4; text-decoration: none; font-weight: 600; margin: 0 .6rem;}
    .foot2 a:hover {text-decoration: underline;}

    /* Mobile */
    @media (max-width: 768px) {
        .block-container {padding-top: 1rem !important;}
        .hero {padding: 2rem 1rem 1.7rem 1rem; border-radius: 18px;}
        .hero h1 {font-size: 1.9rem;}
        .strip, .steps, .strip4 {grid-template-columns: 1fr;}
        .stat .n {font-size: 2.1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()


@st.cache_data
def load_thumb(path, size=(600, 400)):
    return ImageOps.pad(Image.open(path).convert("RGB"), size,
                        method=Image.LANCZOS, color=(14, 24, 35))


@st.cache_data
def load_video_thumb(path, size=(960, 540)):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return ImageOps.pad(img, size, method=Image.LANCZOS, color=(14, 24, 35))


def show_centered(rgb_array):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb_array)).save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<div class="center-img"><img src="data:image/jpeg;base64,{b64}"></div>',
        unsafe_allow_html=True,
    )


def is_bad(name):
    n = name.lower()
    return "without" in n or n.startswith("no") or "no_" in n or "no " in n


def draw_boxes(frame_bgr, result):
    out = frame_bgr.copy()
    t = max(2, round(max(out.shape[:2]) / 250))
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        color = RED if is_bad(model.names[int(c)]) else GREEN
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, t)
    return out


def show_stats(counts, suffix=""):
    cols = st.columns(len(model.names), gap="medium")
    bad_total = 0
    for col, (cid, name) in zip(cols, model.names.items()):
        n = counts.get(cid, 0)
        bad = is_bad(name)
        if bad:
            bad_total += n
        col.markdown(
            f'<div class="stat {"bad" if bad else "good"}">'
            f'<div class="n">{n}</div><div class="l">{name}{suffix}</div></div>',
            unsafe_allow_html=True,
        )
    if not counts:
        st.markdown('<div class="banner warn">Nothing detected.</div>', unsafe_allow_html=True)
    elif bad_total:
        st.markdown(f'<div class="banner warn">⚠️ Without helmet detected ({bad_total})</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="banner ok">✅ Everyone is wearing a helmet</div>',
                    unsafe_allow_html=True)


def process_video(path, progress):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w, h = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, MAX_WIDTH / w)
    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    limit = min(total, int(fps * MAX_SECONDS))

    raw = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    out = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    t0 = time.time()
    peak = Counter()
    i = 0
    while i < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (ow, oh))
        r = model.predict(frame, conf=CONF, iou=IOU, verbose=False)[0]
        for k, v in Counter(int(x) for x in r.boxes.cls).items():
            peak[k] = max(peak[k], v)
        out.write(draw_boxes(frame, r))
        i += 1
        left = int((time.time() - t0) / i * (limit - i))
        progress.progress(i / limit, text=f"Processing video... about {left}s left")
    cap.release()
    out.release()

    final = raw.replace(".mp4", "_h264.mp4")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw, "-vcodec", "libx264",
         "-pix_fmt", "yuv420p", final],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    with open(final, "rb") as f:
        data = f.read()
    os.remove(raw)
    os.remove(final)
    return data, peak


st.markdown(
    """
    <div class="hero">
      <span class="badge">AI-POWERED SAFETY DETECTION</span>
      <h1>🪖 Helmet Detection</h1>
      <p>Identify riders with and without helmets in images and videos.</p>
      <span class="chip">YOLO11</span><span class="chip">Computer Vision</span>
      <span class="chip">Road Safety</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.space()
st.markdown(
    f"""
    <div class="strip">
      <div class="info"><div class="k">MODEL</div><div class="v">YOLO11n</div></div>
      <div class="info"><div class="k">CLASSES</div><div class="v">{len(model.names)}</div></div>
      <div class="info"><div class="k">INPUT</div><div class="v">Image · Video</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.space()
tab_img, tab_vid = st.tabs(["Image", "Video"])

with tab_img:
    st.space()
    file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"], key="img")

    fkey = f"{file.name}-{file.size}" if file else None
    if fkey != st.session_state.get("last_file"):
        st.session_state["last_file"] = fkey
        st.session_state.pop("example", None)

    img = None
    if "example" in st.session_state:
        img = Image.open(st.session_state["example"]).convert("RGB")
        if st.button("← Back"):
            st.session_state.pop("example")
            st.rerun()
    elif file:
        img = Image.open(file).convert("RGB")
    st.space()

    if img is None:
        st.markdown("#### How it works")

        st.markdown(
            """
            <div class="steps">
              <div class="step"><div class="ic i1">📤</div><h4>Upload</h4>
                <p>Add a photo or a short video of riders.</p></div>
              <div class="step"><div class="ic i2">🧠</div><h4>Detect</h4>
                <p>YOLO11 finds every rider and checks for a helmet.</p></div>
              <div class="step"><div class="ic i3">✅</div><h4>Review</h4>
                <p>See bounding boxes and a helmet / no-helmet count.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if EXAMPLES:
            st.space()

            st.markdown("#### Try an example")
            cols = st.columns(len(EXAMPLES), gap="medium")
            for i, (col, path) in enumerate(zip(cols, EXAMPLES)):
                col.image(load_thumb(path), width="stretch")

                if col.button("Use this image", key=f"ex{i}", width="stretch"):
                    st.session_state["example"] = path
                    st.rerun()
    else:
        st.space()
        with st.spinner("Detecting..."):
            result = model.predict(img, conf=CONF, iou=IOU, verbose=False)[0]
        shown = draw_boxes(np.array(img)[..., ::-1], result)[..., ::-1]
        show_centered(shown)
        show_stats(Counter(int(c) for c in result.boxes.cls))


with tab_vid:
    st.space()
    vfile = st.file_uploader("Upload a video (MP4, MOV, AVI or MKV)", type=["mp4", "mov", "avi", "mkv"], key="vid")
    st.caption(f"Only the first {MAX_SECONDS} seconds are processed. This can take up to a minute.")

    vfkey = f"{vfile.name}-{vfile.size}" if vfile else None
    if vfkey != st.session_state.get("last_vfile"):
        st.session_state["last_vfile"] = vfkey
        st.session_state.pop("example_video", None)

    src, key, is_temp = None, None, False
    if "example_video" in st.session_state:
        src = st.session_state["example_video"]
        key = f"example-{src}"
        if st.button("← Back", key="vid_back"):
            st.session_state.pop("example_video")
            st.rerun()
    elif vfile:
        key = f"{vfile.name}-{vfile.size}"
        if st.session_state.get("vid_key") != key:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(vfile.name)[1], delete=False) as tmp:
                tmp.write(vfile.read())
                src = tmp.name
            is_temp = True
    st.space()

    if key is None:
        if VIDEO_EXAMPLES:
            st.markdown("#### Try an example")
            st.space()
            cols = st.columns([1, 3, 1], gap="medium")
            path = VIDEO_EXAMPLES[0]
            thumb = load_video_thumb(path)
            if thumb is not None:
                cols[1].image(thumb, width="stretch")
            if cols[1].button("Use this video", key="exv0", width="stretch"):
                st.session_state["example_video"] = path
                st.rerun()
    else:
        if src and st.session_state.get("vid_key") != key:
            bar = st.progress(0.0, text="Starting...")
            data, peak = process_video(src, bar)
            bar.empty()
            if is_temp:
                os.remove(src)
            st.session_state.update(vid_key=key, vid_data=data, vid_peak=peak)
        st.video(st.session_state["vid_data"])
        show_stats(st.session_state["vid_peak"], " (max in a frame)")

st.space()
st.space()

st.markdown("#### Model performance")

st.markdown(
    "<div class='strip4'>"
    + "".join(
        f"<div class='info'><div class='k'>{k.upper()}</div><div class='v'>{v}</div></div>"
        for k, v in METRICS.items()
    )
    + "</div>",
    unsafe_allow_html=True,
)

st.space()
st.markdown("#### About this project")
st.markdown(
    """
    <div class="about">
      <div class="card">
        <div class="k">OVERVIEW</div>
        <p>Detects whether riders are wearing helmets using a YOLO11n model trained on a
        Roboflow helmet dataset. Built for road-safety use cases such as traffic monitoring.
        Upload an image or a short video, or try one of the examples.</p>
      </div>
      <div class="card">
        <div class="k">BUILT WITH</div>
        <span class="tag">Python</span><span class="tag">YOLO11</span>
        <span class="tag">Ultralytics</span><span class="tag">OpenCV</span>
        <span class="tag">Streamlit</span>
        <div class="legend">
          <div><span class="dot" style="background:#22c55e"></span>Green box: helmet</div>
          <div><span class="dot" style="background:#ef4444"></span>Red box: no helmet</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="foot2">
      <a href="https://github.com/AnishaSinha01" target="_blank">GitHub</a>
      <a href="https://www.linkedin.com/in/anisha-sinha-3abb84326/" target="_blank">LinkedIn</a>
      <div style="margin-top:.8rem">Built by Anisha Sinha · YOLO11 · Ultralytics · Streamlit</div>
    </div>
    """,
    unsafe_allow_html=True,
)