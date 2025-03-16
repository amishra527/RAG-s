create a conda env 

conda create --name llm python=3.10 -y

then install using pip the requirements_IR.txt file for the UI or running bith API al togeather 

run uvicorn infrence_api:app --port=10002 

to view the API

localhost:port_number/docs