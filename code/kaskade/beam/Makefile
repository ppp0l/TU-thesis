include /data/numerik/people/pvillani/kaskade7/Makefile.Local
include $(KASKADE7)/Makefile.Rules

ALLLIBS = $(KASKADELIB) $(DUNELIB) $(UGLIB) $(BOOSTLIB) $(UMFPACKLIB) $(BLASLIB) $(FTNLIB) $(NUMALIB) $(MUMPSLIB) 

INCLUDE = $(DUNEINC) $(UGINC) $(FUSIONINC) $(BOOSTINC) $(UMFPACKINC) $(MUMPSINC) $(HYPREINC) $(KASKADEINC) 

adaBeam: adaBeam.o $(KASKADE7)/libs/libkaskade.a
	$(CXX) $(FLAGS) $< $(KASKADE7)/libs/umfpack_solve.o $(ALLLIBS) $(INCLUDE) $(LINKFLAGS) -o $@
