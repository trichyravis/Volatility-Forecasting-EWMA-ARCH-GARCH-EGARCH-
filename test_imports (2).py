"""
Diagnostic test app to verify all imports work correctly.
Run this first to identify any missing dependencies.
"""

import streamlit as st

st.title("Dependency Diagnostic Test")

# Test each import individually
imports_status = {}

st.write("Testing imports...")

# 1. Streamlit
try:
    import streamlit
    imports_status['streamlit'] = f"✅ {streamlit.__version__}"
except Exception as e:
    imports_status['streamlit'] = f"❌ {str(e)}"

# 2. Pandas
try:
    import pandas as pd
    imports_status['pandas'] = f"✅ {pd.__version__}"
except Exception as e:
    imports_status['pandas'] = f"❌ {str(e)}"

# 3. NumPy
try:
    import numpy as np
    imports_status['numpy'] = f"✅ {np.__version__}"
except Exception as e:
    imports_status['numpy'] = f"❌ {str(e)}"

# 4. Matplotlib
try:
    import matplotlib
    import matplotlib.pyplot as plt
    imports_status['matplotlib'] = f"✅ {matplotlib.__version__}"
except Exception as e:
    imports_status['matplotlib'] = f"❌ {str(e)}"

# 5. SciPy
try:
    from scipy import stats
    import scipy
    imports_status['scipy'] = f"✅ {scipy.__version__}"
except Exception as e:
    imports_status['scipy'] = f"❌ {str(e)}"

# 6. yfinance
try:
    import yfinance as yf
    imports_status['yfinance'] = f"✅ Installed"
except Exception as e:
    imports_status['yfinance'] = f"❌ {str(e)}"

# 7. arch
try:
    from arch import arch_model
    import arch
    imports_status['arch'] = f"✅ {arch.__version__}"
except Exception as e:
    imports_status['arch'] = f"❌ {str(e)}"

# Display results
st.subheader("Import Status:")
for package, status in imports_status.items():
    st.write(f"**{package}:** {status}")

# Check if all imports succeeded
all_ok = all("✅" in status for status in imports_status.values())

if all_ok:
    st.success("🎉 All dependencies are installed correctly!")
    st.info("You can now run the main app.py")
else:
    st.error("⚠️ Some dependencies are missing or failed to import")
    st.write("**Next steps:**")
    st.write("1. Check your requirements.txt file")
    st.write("2. Ensure all packages are listed")
    st.write("3. Reboot the app in Streamlit Cloud")

# Show Python and system info
st.subheader("System Information:")
import sys
import platform

st.write(f"**Python version:** {sys.version}")
st.write(f"**Platform:** {platform.platform()}")
st.write(f"**System:** {platform.system()}")
