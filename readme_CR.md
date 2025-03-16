create a conda env 

conda create --name marker python=3.11 -y

then install using pip the requirements_CR.txt file for the UI or running bith API al togeather 

run uvicorn name_of_convert_file:app --port=10001   

to view the API

localhost:port_number/docs