# Dime v6

Dime v6 es un modelo causal de 12.691.584 parámetros entrenado desde pesos aleatorios para traducir peticiones en español a consultas Bash y expresiones regulares Python. Incluye ocho bloques, dimensión oculta 384, seis cabezas de consulta y dos de clave/valor, tokenización por bytes con 266 símbolos y contexto de 512 tokens.

## Entrenamiento y selección

El corpus sintético v6 contiene 94.943 ejemplos de entrenamiento repartidos en 47 familias. El muestreo asignó un 50 % a Bash, un 40 % a regex y un 10 % a lecciones de copia literal. La corrida completó 4.451 actualizaciones. El checkpoint incluido es el del paso 3.400, elegido por exactitud macro en un selector de 131 peticiones de validación, con la pérdida como desempate.

## Evaluación reservada

El test se evaluó sobre el checkpoint congelado. Cada salida debía coincidir exactamente con su referencia y terminar con el token de fin de respuesta. Las métricas corresponden a la generación directa del modelo.

| Medición | Resultado |
|---|---:|
| Coincidencia exacta total | 700/719 (97,36 %) |
| Bash | 392/410 (95,61 %) |
| Regex | 308/309 (99,68 %) |
| Copia de archivos o directorios dentro de Bash | 392/410 (95,61 %) |

Los desacuerdos se concentraron en la copia de un mismo nombre nuevo en 18 peticiones y en una cantidad en una expresión regular. El corpus cubre consultas de alcance acotado. Las peticiones libres pueden salirse de las operaciones y formatos entrenados.

En la RTX 3080 Laptop, el percentil 95 del tiempo hasta el primer token fue de 15,72 ms en 50 mediciones CUDA/BF16 con el modelo cargado y caliente. La invocación de una sola consulta también incluye el arranque de Python y la carga del modelo.

Los hashes de los pesos y el tokenizador están registrados en [SHA256SUMS](SHA256SUMS). La [instalación y los ejemplos](README.md#instalación) usan ambos archivos incluidos en este repositorio.
