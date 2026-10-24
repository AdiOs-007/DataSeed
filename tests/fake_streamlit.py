"""A tiny recording stand-in for Streamlit so the app script can be executed in tests without a browser.

It is NOT a faithful Streamlit emulator: it checks that the app logic runs and reaches the right states."""
import sys, types, io, importlib
from unittest.mock import MagicMock
class Session(dict):
    __getattr__ = lambda s,k: s[k]
    def __setattr__(s,k,v): s[k]=v
class Up(io.BytesIO):
    name="s.csv"; size=0
def make(clicks, mode_idx, upload=None, text=""):
    st=types.ModuleType("streamlit"); st.session_state=Session(); calls=[]
    def rec(name, ret=None):
        def f(*a,**k):
            calls.append((name,a[0] if a else None)); return ret(*a,**k) if callable(ret) else ret
        return f
    for n in "title caption header subheader markdown write text info warning error success divider json set_page_config".split(): setattr(st,n,rec(n))
    st.dataframe=rec("dataframe"); st.metric=rec("metric"); st.download_button=rec("download",False)
    st.button=lambda label,*a,**k: (calls.append(("button",label)) or label in clicks)
    st.checkbox=lambda l,value=False,**k: value
    st.number_input=lambda l,value=0,**k: 600
    st.text_input=lambda l,value="",**k: value
    st.text_area=lambda l,value="",**k: (text if k.get("key")=="description" else value)
    st.radio=lambda l,opts,**k: opts[mode_idx] if "input method" in l else opts[0]
    st.selectbox=lambda l,opts,**k: opts[0]
    st.file_uploader=lambda l,**k: upload if k.get("key")=="sample_file" else None
    st.spinner=lambda *a,**k: MagicMock(); 
    class Ctx:
        def __enter__(s): return MagicMock()
        def __exit__(s,*a): return False
    st.expander=lambda *a,**k: Ctx(); st.tabs=lambda labels: [Ctx() for _ in labels]
    st.columns=lambda n,**k: [MagicMock(button=lambda label,*a,**kk: label in clicks, metric=lambda *a,**kk:None, download_button=lambda *a,**kk:None,
                                         number_input=lambda l,value=0,**kk:600, text_input=lambda l,value="",**kk:value, checkbox=lambda l,value=False,**kk:value) for _ in range(n if isinstance(n,int) else len(n))]
    st.sidebar=MagicMock(radio=lambda l,opts,**k: opts[mode_idx], markdown=lambda *a,**k:None)
    return st, calls
def run(clicks, mode_idx, upload=None, text="", state=None):
    """Execute dataseed/streamlit_app.py against the recording stub; returns (stub, recorded calls)."""
    st, calls = make(clicks, mode_idx, upload, text)
    if state:
        st.session_state.update(state)
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = st
    sys.modules.pop("dataseed.streamlit_app", None)
    try:
        importlib.import_module("dataseed.streamlit_app")
    finally:
        sys.modules.pop("dataseed.streamlit_app", None)
        if saved is not None:
            sys.modules["streamlit"] = saved
        else:
            sys.modules.pop("streamlit", None)
    return st, calls
