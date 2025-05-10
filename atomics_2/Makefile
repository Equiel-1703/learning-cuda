# === Variables ===
SRC_DIRS := include_src
HIP_DIR := hip
OBJ_DIR := obj
BIN_DIR := bin
COMMON_OBJS_FOLDER = ../common/obj

# Include directories
INCLUDE_DIRS := include

# Find all .cpp and .cu files in the source directories
SRCS := $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cpp $(dir)/*.cu))
SRCS += $(wildcard *.cpp *.cu)

# Find all .hpp files in the source include directories (we need them to hipify)
# Note: This assumes that all .hpp files are in the include directory
INCLUDES_FILES := $(foreach dir,$(INCLUDE_DIRS),$(wildcard $(dir)/*.hpp))

# Replace .cpp/.cu by .o and prepend obj/ (keep path structure)
OBJS := $(patsubst %, $(OBJ_DIR)/%, $(patsubst %.cpp,%.o,$(patsubst %.cu,%.o,$(SRCS))))

COMMON_OBJS = $(wildcard $(COMMON_OBJS_FOLDER)/*.o)

# Replace .cpp/.cu by .hip and prepend hip/ (keep path structure)
HIPS := $(patsubst %, $(HIP_DIR)/%, $(patsubst %.cpp,%.hip,$(patsubst %.cu,%.hip,$(SRCS))))
HIPS_INCLUDES := $(INCLUDES_FILES:%.hpp=$(HIP_DIR)/%.hpp)

# Compiler and flags
CXX := hipcc
CXXFLAGS := -Wall -Wextra -std=c++17
HIPIFY := hipify-clang

# TARGET
TARGET := $(BIN_DIR)/atomics_2.out

# === Rules ===

# Default rule
all: $(HIPS) $(HIPS_INCLUDES) $(OBJS) $(TARGET)

define HIPIFY_RULE
$(HIP_DIR)/%.hip: %.$(1)
	@echo "Hipifying $$< into $$@"
	@mkdir -p $$(dir $$@)
	$$(HIPIFY) $$< -o $$@ -- $$(CXXFLAGS)
	@sed -i 's@\.\./common/include/@../common/hip/@g' $$@
endef

$(eval $(call HIPIFY_RULE,cu))
$(eval $(call HIPIFY_RULE,cpp))

# Rule to hipify all .hpp files
$(HIP_DIR)/%.hpp: %.hpp
	@echo "Hipifying HEADER $< into $@"
	@mkdir -p $(dir $@)
	$(HIPIFY) $< -o $@ -- $(CXXFLAGS)
	@sed -i 's@\.\./common/include/@../common/hip/@g' $@

# Rule to build each .o file
$(OBJ_DIR)/%.o: $(HIP_DIR)/%.hip
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build the final executable
$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $^ $(COMMON_OBJS)

# Clean rule
clean:
	rm -rf $(OBJ_DIR)/*
	rm -rf $(HIP_DIR)/*
	rm -rf $(BIN_DIR)/*

# Optional: pseudo-targets
.PHONY: all clean
