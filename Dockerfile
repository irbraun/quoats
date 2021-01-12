FROM python:3.6
EXPOSE 8501
WORKDIR /quoats

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY data ./data
COPY documentation ./documentation
COPY images ./images
COPY models ./models
COPY ontologies ./ontologies
COPY stored ./stored
COPY lib ./lib
RUN pip install lib/oats-0.0.1-py3-none-any.whl

COPY . .
CMD streamlit run main.py