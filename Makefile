CXXFLAGS=-O3

nbody-gpu: nbody-gpu.cpp
	g++ -O3 nbody-gpu.cpp -o nbody-gpu

solar.out: nbody-gpu
	date
	./nbody-gpu planet 200 5000000 10000 > solar.out # maybe a minutes
	date

solar.pdf: solar.out
	python3 plot.py solar.out solar.pdf 1000 

random.out: nbody-gpu
	date
	./nbody-gpu 1000 1 10000 100 > random.out # maybe 5 minutes
	date
