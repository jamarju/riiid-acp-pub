# Generate .txz files for kaggle dataset:

# set EXT to tgz/txz and COMPTOOL to pigz/xz, DO NOT LEAVE SPACES AFTER THE VALUES
TARGET   := to_upload
EXT      := tgz
COMPTOOL := pigz

# By including dir names in the resources, this will work with both additions/modifications and deletions:
target := to_upload
resources := $(shell find root -print)

.PHONY: all clean
all:  $(TARGET)/resources.$(EXT)

# Local targets

$(TARGET)/resources.$(EXT): $(resources)
	mkdir -p $(TARGET)
	tar -cf - -C root . |$(COMPTOOL) >$@

clean:
	rm -r $(TARGET)

