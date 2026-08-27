## 1. How does shifting your digit toward a corner alter the confidence distribution for each architecture?

Al mover el número hacia una esquina, la confianza de los modelos puede cambiar porque el dibujo ya no aparece en la misma posición que la mayoría de los ejemplos usados durante el entrenamiento. El MLP probablemente es el que más se puede afectar, porque convierte la imagen en una lista de píxeles y cada posición tiene un peso específico. Entonces, si el número cambia de lugar, también cambia mucho la entrada que recibe.

La CNN debería resistir mejor este tipo de desplazamientos porque busca patrones como bordes, líneas y curvas, aunque estén en posiciones distintas. Aun así, en mi prueba la CNN confundió el 5 con un 3, así que también puede fallar si el dibujo cambia demasiado o si pierde detalles al reducirse a 28 × 28 píxeles.

La RNN también puede cambiar su confianza porque analiza la imagen como una secuencia de filas. Si el número se mueve, esa secuencia cambia. En mi caso, fue la única arquitectura que reconoció correctamente el número 5, lo cual me mostró que el comportamiento de cada modelo puede variar bastante dependiendo de cómo esté dibujada la imagen.

## 2. What implications does CNN translation robustness have for computer vision systems deployed in uncontrolled environments?

La robustez de una CNN ante desplazamientos es importante porque en situaciones reales los objetos casi nunca aparecen exactamente en el centro de una imagen. Por ejemplo, una persona, un automóvil o una señal pueden aparecer en diferentes posiciones dependiendo de la cámara o del momento.

Las CNN tienen ventaja porque pueden reconocer características locales aunque cambien de posición. Esto ayuda a que los sistemas de visión por computadora sean más flexibles en ambientes reales, donde no se puede controlar completamente cómo aparece cada objeto.

Sin embargo, entendí que esta capacidad no es perfecta. En mi prueba, la CNN no reconoció correctamente el 5. Esto me hace pensar que además de la arquitectura, también influye mucho el tipo de entrenamiento, la cantidad de datos y la variedad de ejemplos que recibe el modelo.
