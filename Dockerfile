# Use an official Python runtime as a base iamge

FROM tensorflow/tensorflow:1.10.0-gpu

RUN apt-get update -y && apt-get install -y \
	libgeos-dev \
	python-pip \
	python-tk

RUN pip install numpy
RUN pip install keras

RUN mkdir /RNN_niry

# Add files to the dockerfile
COPY rnn_functions.py /RNN_niry/rnn_functions.py
COPY mrf_recon_rnn_tr_init.py /RNN_niry/mrf_recon_rnn_tr_init.py
#COPY mrf_recon_rnn_par_search.py /RNN_niry/mrf_recon_rnn_par_search.py
#COPY mrf_recon_rnn_tr_series_len.py /RNN_niry/mrf_recon_rnn_tr_series_len.py
#COPY mrf_recon_rnn_fc_init.py /RNN_niry/mrf_recon_rnn_fc_init.py
#COPY mrf_recon_rnn_fc_series_len.py /RNN_niry/mrf_recon_rnn_fc_series_len.py
COPY dic.py /RNN_niry/dic.py

WORKDIR "/RNN_niry"

# Make port 80 available to the world outside this container
EXPOSE 6006

# Run when the container launches

#CMD exec python mrf_recon_rnn_tr_init.py
#CMD exec python mrf_recon_rnn_par_search.py
CMD exec python mrf_recon_rnn_tr_series_len.py
#CMD exec python mrf_recon_rnn_fc_init.py
#CMD exec python mrf_recon_rnn_fc_series_len.py
