zip:
	zip -r result/gan/gen.zip result/gan/gen/
	zip -r result/gan/disc.zip result/gan/disc/



all:
	rm -rf result/gan/gen/*
	rm -rf result/gan/disc/*

	mv result/gan/gen_*.pth result/gan/gen/
	mv result/gan/disc_*.pth result/gan/disc/

	make zip