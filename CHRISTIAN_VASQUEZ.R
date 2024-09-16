#-----------------------------------#
#    SEGUNDA PR√ÅCTICA CALIFICADA    # 
#      DETECCI√ìN DE ANOMAL√çAS       #
#       REGLAS DE ASOCIACI√ìN        #
#     Mg. Jes√∫s Salinas Flores      # 
#     jsalinas@lamolina.edu.pe      #
#-----------------------------------#

# Para limpiar el workspace, por si hubiera algun dataset 
# o informaci√≥n cargada
rm(list = ls())
graphics.off()

# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

# Otras opciones
options(scipen = 999)    # Eliminar la notaci√≥n cient√≠fica
options(digits = 3)      # N√∫mero de decimales


#---------------------------#
# I. Detecci√≥n de Anomal√≠as #
#---------------------------#
browseURL("http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/")

# Dataset information
# The original Cardiotocography (Cardio) dataset from UCI 
# machine learning repository consists of measurements of 
# fetal heart rate (FHR) and uterine contraction (UC) features
# on cardiotocograms classified by expert obstetricians. 
# This is a classification dataset, where the classes are
# normal, suspect, and pathologic. For outlier detection, The
# normal class formed the inliers, while the pathologic
# (outlier) class is downsampled to 176 points. 
# The suspect class is discarded.

# TraducciÛn

# El conjunto de datos original de CardiotocografÌa (Cardio) del repositorio de 
# aprendizaje autom·tico de la UCI consiste en mediciones de la frecuencia 
# cardÌaca fetal (FCF) y caracterÌsticas de la contracciÛn uterina (UC) en 
# cardiotocografÌas clasificadas por obstetras expertos. 
# Se trata de un conjunto de datos de clasificaciÛn, en el que las clases 
# son normal, sospechosa y patolÛgica. Para la detecciÛn de valores atÌpicos, 
# la clase normal formÛ los inliers, mientras que la clase patolÛgica (outlier) 
# se reduce a 176 puntos. 
# La clase sospechosa se descarta.

library(R.matlab)
cardio_mat  <- readMat("cardio.mat")
df_cardio   <- as.data.frame(cardio_mat$X)
df_cardio$y <- as.factor(cardio_mat$y)

levels(df_cardio$y)
levels(df_cardio$y) <- c("Normal","PatolÛgica")

str(df_cardio)

contrasts(df_cardio$y)

table(df_cardio$y)
prop.table(table(df_cardio$y))*100

# Particione la muestra en train (80%) y de evaluaciÛn (20%). 
# Use set.seed(2021) donde corresponda
# Aplique el algoritmo Isolation Forest y compare su
# sensibilidad, especificidad y accuracy balanceado vs. 
# un modelo de RegresiÛn LogÌstica con umbral 0.5 y un 
# umbral Ûptimo.
# Indique el que elegirÌa. Justifique su respuesta
# (10 puntos)

#  Comparacion modelos           Umbral  Sensibilidad  Especificidad  Accuracy Balanceado
# 1. Logistica     
# 2. Logistica con umbral optimo 
# 3. IsoFor 
# 4. IsoFor con umbral optimo 

# ResoluciÛn

# Paquetes
library(pacman)
p_load(tictoc, data.table, outliers, ggplot2, caret,
       caTools, pROC, isofor,dplyr)

datos.a <- df_cardio

set.seed(2021) 
index         <- createDataPartition(datos.a$y, 
                                     p=0.8, 
                                     list=FALSE)
train   <- datos.a[ index, ]
testing <- datos.a[-index, ]

# Verificando que se mantenga la proporciÛn original
round(prop.table(table(datos.a$y))*100,2)
round(prop.table(table(train$y))*100,2)
round(prop.table(table(testing$y))*100,2)

#----------------------------------------#
# 1. Entrenando con RegresiÛn LogÌstica  #
#----------------------------------------#

# 1.1 Modelo logÌstico con todas las variables ----------------
modelo_rl  <- glm(y ~ . , 
                  family=binomial,
                  data=train)

modelo_rl

# Prediciendo la probabilidad
testing$proba.pred <- predict(modelo_rl,testing,type="response")
head(testing$proba.pred)

# Prediciendo la clase con punto de corte (umbral) igual a 0.5
testing$clase.pred <- factor(ifelse(testing$proba.pred >= 0.5, 
                                    "PatolÛgica","Normal"))
head(testing$clase.pred)

#------------------------------------------#
# 2. Entrenar con DetecciÛn de AnomalÌas   #
#      Algoritmo Isolation Forest          #
#------------------------------------------#

# 2.1 Modelo con Isolation Forest -----------------------------
set.seed(2021)
modelo_iso_forest <- iForest(train%>%select(-y), # No considerar al target
                             nt  = 100,   # n˙mero de ·rboles
                             phi = 1000)  # TamaÒo de la submuestra 
# para construir cada ·rbol

# Determinar el Anomaly Score 
testing$iso_score <- predict(modelo_iso_forest,testing%>%select(-y))
head(testing$iso_score)
summary(testing$iso_score)

ggplot(testing) + aes(x = iso_score) +
  geom_histogram(color = "gray40") +
  geom_vline(
    xintercept = quantile(testing$iso_score,seq(0,1,0.05)),
    color      = "red",
    linetype   = "dashed") +
  labs(title = "Distribuci√≥n de los scores del Isolation Forest") +
  theme_bw()


# Boxplot del Anomaly Score vs. Target
ggplot(testing, aes(y,iso_score)) + 
  geom_boxplot(fill=c("cadetblue","firebrick1")) 

# Calcular el umbral m·s alto para el iso_score
high_iso <- quantile(testing$iso_score, probs = 0.95)  
high_iso

# Determinar la clase predicha usando el Binary Isolation Score
testing$binary_iso2 <- as.numeric(testing$iso_score >= high_iso)
head(testing$binary_iso2)
tail(testing$binary_iso2)

summary(testing$binary_iso2)

testing$binary_iso2         <- factor(testing$binary_iso2)
levels(testing$binary_iso2) <- c("Normal","PatolÛgica")

#-----------------------------------#
# 3. Mejorando indicadores usando   #
#     punto de corte √≥ptimo        #
#     Algoritmo Isolation Forest    #
#-----------------------------------#

# 3.1 Obteniendo el umbral Ûptimo en Isolation Forest ---------
modelroc1 <- roc(testing$y,testing$iso_score) # Clase Real vs Proba predicha
plot(modelroc1, 
     print.auc=TRUE, 
     print.thres=TRUE,
     auc.polygon=TRUE,
     col="blue",
     auc.polygon.col="lightblue",
     xlab="1 - Especificidad", 
     ylab="Sensibilidad")

umbral1 <- pROC::coords(modelroc1, "best")$threshold
umbral1

# Cambiando el punto de corte (umbral Ûptimo)
testing$binary_iso3 <- as.numeric(testing$iso_score >= umbral1)
head(testing$binary_iso3)
tail(testing$binary_iso3)

summary(testing$binary_iso3)

testing$binary_iso3         <- factor(testing$binary_iso3)
levels(testing$binary_iso3) <- c("Normal","PatolÛgica")

#---------------------------------#
# 4. Mejorando indicadores usando #
#    punto de corte Ûptimo        #
#    RegresiÛn LogÌstica          #
#---------------------------------#

# 4.1 Obteniendo el umbral Ûptimo en RegresiÛn LogÌstica ------
modelroc2 <- roc(testing$y,testing$proba.pred) # Clase Real vs Proba predicha
plot(modelroc2, 
     print.auc=TRUE, 
     print.thres=TRUE,
     auc.polygon=TRUE,
     col="blue",
     auc.polygon.col="lightblue",
     xlab="1 - Especificidad", 
     ylab="Sensibilidad")

umbral2 <- pROC::coords(modelroc2, "best")$threshold
umbral2

# Cambiando el punto de corte (umbral Ûptimo)
testing$clase.pred.u <- as.numeric(testing$proba.pred >= umbral2)
head(testing$clase.pred.u)
tail(testing$clase.pred.u)

summary(testing$clase.pred.u)

testing$clase.pred.u         <- factor(testing$clase.pred.u)
levels(testing$clase.pred.u) <- c("Normal","PatolÛgica")

# 5. Comparando los cuatro modelos ----------------------------

result1 <- caret::confusionMatrix(testing$clase.pred,
                                  testing$y,
                                  positive="PatolÛgica")

colAUC(testing$proba.pred,testing$y,plotROC = TRUE) -> auc1

result2 <- caret::confusionMatrix(testing$binary_iso2,
                                  testing$y,
                                  positive="PatolÛgica")

colAUC(testing$iso_score,testing$y,plotROC = TRUE) -> auc2

result3 <- caret::confusionMatrix(testing$binary_iso3,
                                  testing$y,
                                  positive="PatolÛgica")

colAUC(testing$iso_score,testing$y,plotROC = TRUE) -> auc3

result4 <- caret::confusionMatrix(testing$clase.pred.u,
                                  testing$y,
                                  positive="PatolÛgica")

colAUC(testing$proba.pred,testing$y,plotROC = TRUE) -> auc4

modelos  <- c("LogÌstica",
              "Logistica-umbral ”ptimo",
              "IsoFor-umbral 5%",
              "IsoFor-umbral ”ptimo")

umbrales <- c(0.5,umbral2,high_iso,umbral1)

sensibilidad  <- c(result1$byClass["Sensitivity"],
                   result4$byClass["Sensitivity"],
                   result2$byClass["Sensitivity"],
                   result3$byClass["Sensitivity"])

especificidad <- c(result1$byClass["Specificity"],
                   result4$byClass["Specificity"],
                   result2$byClass["Specificity"],
                   result3$byClass["Specificity"])

accuracy      <- c(result1$overall["Accuracy"],
                   result4$overall["Accuracy"],
                   result2$overall["Accuracy"],
                   result3$overall["Accuracy"])

acc_bal       <- c(result1$byClass["Balanced Accuracy"],
                   result4$byClass["Balanced Accuracy"],
                   result2$byClass["Balanced Accuracy"],
                   result3$byClass["Balanced Accuracy"])

auc           <- c(auc1, auc4, auc2, auc3)

comparacion <- data.frame(modelos,
                          umbrales,
                          sensibilidad,
                          especificidad,
                          accuracy,
                          acc_bal,
                          auc)

comparacion
#                   modelos umbrales sensibilidad especificidad accuracy acc_bal   auc
# 1               LogÌstica    0.500        0.943         0.985    0.981   0.964 0.995
# 2 Logistica-umbral ”ptimo    0.487        0.971         0.985    0.984   0.978 0.995
# 3        IsoFor-umbral 5%    0.517        0.257         0.970    0.902   0.613 0.937
# 4    IsoFor-umbral ”ptimo    0.444        0.943         0.849    0.858   0.896 0.937

# CONCLUSI”N

# Me quedarÌa con el modelo de regresiÛn logÌstica con umbral Ûptimo, debido a 
# que tuvo la mayor precisiÛn para detectar a los verdaderos casos de embarazos 
# con problemas patolÛgicos, con un 94.3 % de sensibilidad. 
# Siendo un caso donde se trata de detectar problemas anomalÌas (enfermedades
# cardiacas en bebÈs y contracciones uterinas) en los embarazos, el mayor interÈs
# del modelo ser· detectar casos positivos de la variable predicha, es decir,
# comportamiento patolÛgicos (PatolÛgica) en el embarazo, para que puedan
# recibir estas madres gestantes un tratamiento mÈdico a tiempo.
# Por otro lado, este modelo tiene un umbral de 0.487 el cual est· alejado de 
# los valores extremos (0 o 1) de 1 - Especificidad, por lo tanto, 
# se corre menor riesgo de sobreajuste.

#--------------------------#
# II. Reglas de AsociaciÛn #
#--------------------------#

# El archivo de datos Shopping.txt, contiene informaciÛn 
# socio-demogr·fica de compradores y 10 campos de 
# productos, codificados como 1 o 0 indicando si se ha
# comprado una categorÌa de producto o no.

datos.completos <- read.delim("Shopping.txt")

datos.modelo    <- datos.completos[, 11:15] %>% 
                   mutate_all(as.factor) 

datos.productos <- datos.completos[, c(1:10)]
datos.productos <- datos.productos %>% 
                   mutate_all(as.logical) 

# Use set.seed(2021) donde corresponda.
# Aplique el algoritmo a priori con soporte mÌnimo del 15%,
# confianza mÌnima del 40% y reglas de 2 a 10 productos.
# Encuentre las principales reglas y elimine las reglas 
# redundantes. (2 puntos)
# Ordene las reglas en forma descendente seg˙n el Lift. Luego, 
# use la primera regla de dicho orden e interprete su soporte, 
# confianza y lift (3 puntos)
# Indique cu·l serÌa el patrÛn de los que comprarÌan los 
# productos de dicha primera regla en funciÛn a las variables 
# socio-demogr·ficas. Justifique su respuesta.
# (5 puntos)

# Paquetes 
library(pacman)
p_load(arules, arulesViz, RColorBrewer, dplyr, ggplot2, 
       foreign, colorspace, rpart, rpart.plot) 

# A) Aplique el algoritmo a priori con soporte mÌnimo del 15%,
# confianza mÌnima del 40% y reglas de 2 a 10 productos.
# Encuentre las principales reglas y elimine las reglas 
# redundantes. (2 puntos)

# Paso 1: Encontrando las reglas de asociaciÛn ----------------
library(arules)
reglas <- apriori(data = datos.productos,
                  parameter = list(support = 0.15,
                                   confidence = 0.40,
                                   minlen = 2, maxlen = 10,
                                   # Se especifica que se creen reglas
                                   target = "rules"))
reglas

# B) Encuentre las principales reglas y elimine las reglas 
# redundantes. (2 puntos)

# 2. Reglas redundantes
inspect(reglas[is.redundant(reglas)])

length(reglas[is.redundant(reglas)])
# No existen reglas redundantes

# Encontrando reglas no redundantes
inspect(reglas[!is.redundant(reglas)])

#      lhs                         rhs                      support confidence coverage lift  count
# [1]  {Alcohol}                => {Alimentos.congelados}   0.230   0.584      0.394    1.452 181  
# [2]  {Alimentos.congelados}   => {Alcohol}                0.230   0.573      0.402    1.452 181  
# [3]  {Alcohol}                => {Alimentos.en.Conservas} 0.173   0.439      0.394    0.963 136  
# [4]  {Alcohol}                => {Productos.de.Panaderia} 0.215   0.545      0.394    1.272 169  
# [5]  {Productos.de.Panaderia} => {Alcohol}                0.215   0.501      0.429    1.272 169  
# [6]  {Alcohol}                => {Aperitivos}             0.219   0.555      0.394    1.169 172  
# [7]  {Aperitivos}             => {Alcohol}                0.219   0.461      0.475    1.169 172  
# [8]  {Alcohol}                => {Alimentos.Listos}       0.212   0.539      0.394    1.094 167  
# [9]  {Alimentos.Listos}       => {Alcohol}                0.212   0.432      0.492    1.094 167  
# [10] {Alimentos.congelados}   => {Alimentos.en.Conservas} 0.207   0.516      0.402    1.133 163  
# [11] {Alimentos.en.Conservas} => {Alimentos.congelados}   0.207   0.455      0.455    1.133 163  
# [12] {Alimentos.congelados}   => {Productos.de.Panaderia} 0.221   0.551      0.402    1.284 174  
# [13] {Productos.de.Panaderia} => {Alimentos.congelados}   0.221   0.516      0.429    1.284 174  
# [14] {Alimentos.congelados}   => {Aperitivos}             0.214   0.532      0.402    1.120 168  
# [15] {Aperitivos}             => {Alimentos.congelados}   0.214   0.450      0.475    1.120 168  
# [16] {Alimentos.congelados}   => {Alimentos.Listos}       0.211   0.525      0.402    1.067 166  
# [17] {Alimentos.Listos}       => {Alimentos.congelados}   0.211   0.429      0.492    1.067 166  
# [18] {Alimentos.en.Conservas} => {Productos.de.Panaderia} 0.228   0.500      0.455    1.166 179  
# [19] {Productos.de.Panaderia} => {Alimentos.en.Conservas} 0.228   0.531      0.429    1.166 179  
# [20] {Alimentos.en.Conservas} => {Aperitivos}             0.224   0.492      0.455    1.036 176  
# [21] {Aperitivos}             => {Alimentos.en.Conservas} 0.224   0.472      0.475    1.036 176  
# [22] {Alimentos.en.Conservas} => {Alimentos.Listos}       0.216   0.475      0.455    0.964 170  
# [23] {Alimentos.Listos}       => {Alimentos.en.Conservas} 0.216   0.439      0.492    0.964 170  
# [24] {Productos.de.Panaderia} => {Aperitivos}             0.233   0.543      0.429    1.144 183  
# [25] {Aperitivos}             => {Productos.de.Panaderia} 0.233   0.491      0.475    1.144 183  
# [26] {Productos.de.Panaderia} => {Alimentos.Listos}       0.256   0.596      0.429    1.211 201  
# [27] {Alimentos.Listos}       => {Productos.de.Panaderia} 0.256   0.519      0.492    1.211 201  
# [28] {Aperitivos}             => {Alimentos.Listos}       0.244   0.515      0.475    1.045 192  
# [29] {Alimentos.Listos}       => {Aperitivos}             0.244   0.496      0.492    1.045 192  

length(reglas[!is.redundant(reglas)])

# Existen 29 reglas no redundantes o principales, las cuales son:

#      lhs                         rhs                      
# [1]  {Alcohol}                => {Alimentos.congelados}   
# [2]  {Alimentos.congelados}   => {Alcohol}                
# [3]  {Alcohol}                => {Alimentos.en.Conservas} 
# [4]  {Alcohol}                => {Productos.de.Panaderia} 
# [5]  {Productos.de.Panaderia} => {Alcohol}                
# [6]  {Alcohol}                => {Aperitivos}             
# [7]  {Aperitivos}             => {Alcohol}                
# [8]  {Alcohol}                => {Alimentos.Listos}       
# [9]  {Alimentos.Listos}       => {Alcohol}                
# [10] {Alimentos.congelados}   => {Alimentos.en.Conservas} 
# [11] {Alimentos.en.Conservas} => {Alimentos.congelados}   
# [12] {Alimentos.congelados}   => {Productos.de.Panaderia} 
# [13] {Productos.de.Panaderia} => {Alimentos.congelados}   
# [14] {Alimentos.congelados}   => {Aperitivos}             
# [15] {Aperitivos}             => {Alimentos.congelados}   
# [16] {Alimentos.congelados}   => {Alimentos.Listos}       
# [17] {Alimentos.Listos}       => {Alimentos.congelados}   
# [18] {Alimentos.en.Conservas} => {Productos.de.Panaderia} 
# [19] {Productos.de.Panaderia} => {Alimentos.en.Conservas} 
# [20] {Alimentos.en.Conservas} => {Aperitivos}             
# [21] {Aperitivos}             => {Alimentos.en.Conservas} 
# [22] {Alimentos.en.Conservas} => {Alimentos.Listos}       
# [23] {Alimentos.Listos}       => {Alimentos.en.Conservas} 
# [24] {Productos.de.Panaderia} => {Aperitivos}             
# [25] {Aperitivos}             => {Productos.de.Panaderia} 
# [26] {Productos.de.Panaderia} => {Alimentos.Listos}       
# [27] {Alimentos.Listos}       => {Productos.de.Panaderia} 
# [28] {Aperitivos}             => {Alimentos.Listos}       
# [29] {Alimentos.Listos}       => {Aperitivos}             


# 3. Eliminando reglas redundantes
reglas <- reglas[!is.redundant(reglas)]

# Convirtiendo las reglas a un data frame
rules_df <- as(reglas, "data.frame")

# C) Ordene las reglas en forma descendente seg˙n el Lift. Luego, 
# use la primera regla de dicho orden e interprete su soporte, 
# confianza y lift (3 puntos)

# 4. Ordenando las reglas por el lift en orden descendente

rules_df%>%arrange(desc(lift))
#                                                   rules support confidence coverage  lift count
# 1                   {Alcohol} => {Alimentos.congelados}   0.230      0.584    0.394 1.452   181
# 2                   {Alimentos.congelados} => {Alcohol}   0.230      0.573    0.402 1.452   181
# 3    {Alimentos.congelados} => {Productos.de.Panaderia}   0.221      0.551    0.402 1.284   174
# 4    {Productos.de.Panaderia} => {Alimentos.congelados}   0.221      0.516    0.429 1.284   174
# 5                 {Alcohol} => {Productos.de.Panaderia}   0.215      0.545    0.394 1.272   169
# 6                 {Productos.de.Panaderia} => {Alcohol}   0.215      0.501    0.429 1.272   169
# 7        {Productos.de.Panaderia} => {Alimentos.Listos}   0.256      0.596    0.429 1.211   201
# 8        {Alimentos.Listos} => {Productos.de.Panaderia}   0.256      0.519    0.492 1.211   201
# 9                             {Alcohol} => {Aperitivos}   0.219      0.555    0.394 1.169   172
# 10                            {Aperitivos} => {Alcohol}   0.219      0.461    0.475 1.169   172
# 11 {Alimentos.en.Conservas} => {Productos.de.Panaderia}   0.228      0.500    0.455 1.166   179
# 12 {Productos.de.Panaderia} => {Alimentos.en.Conservas}   0.228      0.531    0.429 1.166   179
# 13             {Productos.de.Panaderia} => {Aperitivos}   0.233      0.543    0.429 1.144   183
# 14             {Aperitivos} => {Productos.de.Panaderia}   0.233      0.491    0.475 1.144   183
# 15   {Alimentos.congelados} => {Alimentos.en.Conservas}   0.207      0.516    0.402 1.133   163
# 16   {Alimentos.en.Conservas} => {Alimentos.congelados}   0.207      0.455    0.455 1.133   163
# 17               {Aperitivos} => {Alimentos.congelados}   0.214      0.450    0.475 1.120   168
# 18               {Alimentos.congelados} => {Aperitivos}   0.214      0.532    0.402 1.120   168
# 19                      {Alcohol} => {Alimentos.Listos}   0.212      0.539    0.394 1.094   167
# 20                      {Alimentos.Listos} => {Alcohol}   0.212      0.432    0.492 1.094   167
# 21         {Alimentos.Listos} => {Alimentos.congelados}   0.211      0.429    0.492 1.067   166
# 22         {Alimentos.congelados} => {Alimentos.Listos}   0.211      0.525    0.402 1.067   166
# 23                   {Aperitivos} => {Alimentos.Listos}   0.244      0.515    0.475 1.045   192
# 24                   {Alimentos.Listos} => {Aperitivos}   0.244      0.496    0.492 1.045   192
# 25             {Alimentos.en.Conservas} => {Aperitivos}   0.224      0.492    0.455 1.036   176
# 26             {Aperitivos} => {Alimentos.en.Conservas}   0.224      0.472    0.475 1.036   176
# 27       {Alimentos.en.Conservas} => {Alimentos.Listos}   0.216      0.475    0.455 0.964   170
# 28       {Alimentos.Listos} => {Alimentos.en.Conservas}   0.216      0.439    0.492 0.964   170
# 29                {Alcohol} => {Alimentos.en.Conservas}   0.173      0.439    0.394 0.963   136

# La primera regla de la lista es:
#                                                   rules support confidence coverage  lift count
# 1                   {Alcohol} => {Alimentos.congelados}   0.230      0.584    0.394 1.452   181
# Soporte: 0.230
# El soporte indica que el 23 % de los clientes han comprado alcohol y alimentos congelados.
# Confianza: 0.584
# La confianza indica que el 58.4 % de clientes que compran alcohol, tambiÈn 
# compran alimentos congelados.
# Lift: 1.452
# El lift indica que los clientes que compran alcohol, son 1.452 veces m·s 
# propensos a comprar alimentos congelados.

# D) Indique cu·l serÌa el patrÛn de los que comprarÌan los 
# productos de dicha primera regla en funciÛn a las variables 
# socio-demogr·ficas. Justifique su respuesta.
# (5 puntos)

# Paso 5: Buscando patrones en una regla usando datos socio-demogr·ficos ----
datos.completos$regla<- ifelse(datos.completos$Alcohol == T &
                                 datos.completos$Alimentos.congelados == T,
                               "Compra", "No_Compra")

datos.completos$regla <- as.factor(datos.completos$regla)

datos.clasificacion <- datos.completos[, c(11:15, 16)]  # 11:15 Variables predictoras, 16 target
str(datos.clasificacion)

# Paso 6. Aplicando un ·rbol de clasificaciÛn (CART) -----------
#         para encontrar reglas (patrones)
library(rpart)
set.seed(2021)
arbol1 <- rpart(regla ~ . ,                     
                data = datos.clasificacion, 
                method = "class")   # "anova" para RegresiÛn

arbol1

# n= 786 
# 
# node), split, n, loss, yval, (yprob)
# * denotes terminal node
# 
# 1) root 786 181 No_Compra (0.2303 0.7697)  
# 2) Hijos=No  256  83 Compra (0.6758 0.3242)  
# 4) Estado.Civil=Casado  ,Divorced ,Separated,Soltero    177  28 Compra (0.8418 0.1582) *
#   5) Estado.Civil=Viudo 79  24 No_Compra (0.3038 0.6962) *
#     3) Hijos=Si 530   8 No_Compra (0.0151 0.9849) *

# Graficando el ·rbol
library(rpart.plot)
rpart.plot(arbol1, digits = -1, type = 2, 
           extra = 101, cex = 0.7, nn = TRUE)

# El algoritmo CART indica que el 23 % de clientes que compran alcohol y alimentos congelados 
# son de estado civil casado, divorciado, separado o soltero y con hijos, pues el
# 84.2 % de clientes con estas caracterÌsticas son los que se han registrado que
# compran alcohol y alimentos congelados a la vez.
# Se concluye que el patron de clientes que compran a la vez alcohol y alimentos
# congelados son adultos de estado civil casado, divorciado, separado o soltero 
# y con hijos.


