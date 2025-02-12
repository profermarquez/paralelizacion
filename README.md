# Contexto
Intentaremos generar un Copo de Nieve de Koch en una ventana con pygame y distribuir su cálculo en varios hilos usando dask o incluso en varias ventanas. Aquí hay dos enfoques interesantes:
- Paralelizando el calculo del copo de nive de Koch
- Paralelizando varias redes neuronales artificiales que aprendan a generara copos de nieve de Koch (mas lento y requiere de recursos de CPU y RAM)

# Crear y activar el entorno virtual
/virtualenv env         /env/Scripts/activate.bat

# Requerimientos
pip install pygame dask

pip install numpy

pip install tensorflow

# Ejecutar
py copo_koch.py
py .\copo_koch_con_redes_neuronales.py
