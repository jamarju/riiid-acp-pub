TARGET = 02_train.py

all: $(TARGET)

%.py: %.ipynb
	jupyter nbconvert --RegexRemovePreprocessor.patterns="['^#noexport']" --to script $<

clean:
	$(RM) $(TARGET)
