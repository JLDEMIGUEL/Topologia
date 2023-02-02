#DEV: docker run -dp 5000:5005 app-topologiadocker run -dp 5000:5005 -w /app-topologia -v "C:\Users\josel\Documents\Universidad\Topologia\Topologia:/app-topologia" app-topologia
#PROD:
FROM python:3.10
EXPOSE 5000
WORKDIR /app-topologia
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
RUN cd app
CMD ["flask", "run", "--host", "0.0.0.0"]