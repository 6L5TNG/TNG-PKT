"""Run TNG PKT from the source root with python -m tngpkt."""
import runpy

if __name__ == "__main__":
    runpy.run_module("tngpkt.neuromod_app", run_name="__main__")
