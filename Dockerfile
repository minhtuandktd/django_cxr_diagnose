FROM python:3.9.13

ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY requirements.txt . 
COPY . .

# install dependencies
RUN apt-get update 
RUN apt install -y libgl1-mesa-glx
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# tell the port number container should expose
EXPOSE 8000

# run command
CMD ["python", "manage.py", "runserver","0.0.0.0:8000"]