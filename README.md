# Proyecto-Final-del-Curso-de-Inteligencia-Artificial
Proyecto Transformer Federado para Análisis de Tráfico de Red

Problema:
Los sistemas de detección de intrusos (IDS) necesitan muchos datos de red para entrenar modelos, pero centralizar datos es lento, costoso y poco seguro.
Solución propuesta:
Un proyecto que entrena un Transformer Federado que analiza tráfico de red distribuido sin compartir los datos crudos, solo el conocimiento.
Ventajas:
Privacidad: los datos nunca salen de cada cliente.
Eficiencia: se envían solo parámetros, no millones de paquetes.
Escalabilidad: más nodos es igual a más inteligencia global.

Metodología:
**Simular 2 o 3 clientes federados (nodos) en una sola máquina.
Usar un dataset público NSL-KDD.
Entrenar un transformer pequeño en cada cliente.
Implementar un servidor de agregación federada 
Mostrar resultados de precisión y eficiencia frente a un modelo centralizado.
**
