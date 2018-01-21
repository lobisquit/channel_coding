all: iters_vs_SNR_per_length iters_vs_SNR_per_rate Pe_vs_SNR_per_length Pe_vs_SNR_per_max_iter Pe_vs_SNR_per_rate plot-phi results_per_max_iter

iters_vs_SNR_per_length: report/plots/iters_vs_SNR_per_length.r
	R -f report/plots/iters_vs_SNR_per_length.r

iters_vs_SNR_per_rate: report/plots/iters_vs_SNR_per_rate.r
	R -f report/plots/iters_vs_SNR_per_rate.r

Pe_vs_SNR_per_length: report/plots/Pe_vs_SNR_per_length.r
	R -f report/plots/Pe_vs_SNR_per_length.r

Pe_vs_SNR_per_max_iter: report/plots/Pe_vs_SNR_per_max_iter.r
	R -f report/plots/Pe_vs_SNR_per_max_iter.r

Pe_vs_SNR_per_rate: report/plots/Pe_vs_SNR_per_rate.r
	R -f report/plots/Pe_vs_SNR_per_rate.r

plot-phi: report/plots/plot-phi.py
	cp report/plots/plot-phi.py .; \
	source venv/bin/activate; \
	python plot-phi.py; \
	rm plot-phi.py; \

results_per_max_iter: report/plots/results_per_max_iter.py
	source venv/bin/activate; \
	cp report/plots/results_per_max_iter.py .; \
	python report/plots/results_per_max_iter.py; \
	rm results_per_max_iter.py; \
	make Pe_vs_SNR_per_max_iter; \
