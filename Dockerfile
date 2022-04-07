FROM python:3.8

WORKDIR .

COPY requirements_cpu.txt requirements_cpu.txt
RUN pip3 install -r requirements_cpu.txt

COPY data data
COPY models models

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]