for  %%f in (*.jpeg *.jpg) do ( 
   java -jar test_neuronet.jar %%f %1
)