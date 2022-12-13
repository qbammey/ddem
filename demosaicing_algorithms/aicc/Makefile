COBJ	= io_png.o
CXXOBJ	= libauxiliar.o libdemosaicking.o demosaicking_ipol.o imdiff_ipol.o

BIN	= demosaicking_ipol imdiff_ipol

hdrdir= -I/usr/local/include/ -I/usr/include/
libdir= -L/usr/local/lib/ -L/usr/lib/           

COPT	= -O3 -funroll-loops -fomit-frame-pointer  -Wall -Wextra  
CFLAGS  += $(COPT) $(hdrdir)

LDFLAGS +=  $(libdir) -lpng

# use openMP with `make OMP=1`
ifdef OMP
CFLAGS	+= -fopenmp
LDFLAGS += -lgomp 
endif

default: $(COBJ) $(CXXOBJ)  $(BIN)

$(COBJ) : %.o : %.c 
	$(CC) -c $(CFLAGS)   $< -o $@

$(CXXOBJ) : %.o : %.cpp 
	$(CXX) -std=c++98 -c $(CFLAGS)   $< -o $@

$(BIN) : % : %.o  io_png.o libauxiliar.o libdemosaicking.o
	$(CXX)  -std=c++98  -o $@  $^ $(LDFLAGS)

.PHONY : clean
clean:
	$(RM) $(COBJ) $(CXXOBJ) ; rm $(BIN)
