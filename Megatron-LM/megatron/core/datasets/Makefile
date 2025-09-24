CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color

# Use the current Python interpreter (overridable: `make PYTHON=$(which python)`) 
PYTHON ?= python

CPPFLAGS += $(shell $(PYTHON) -m pybind11 --includes)

LIBNAME = helpers_cpp
# Determine extension suffix from the active interpreter to avoid ABI mismatches
LIBEXT = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp

default: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
