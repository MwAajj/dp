package dataset;


import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;

public class DatasetManager {

    private String fileName = "src/main/resources/files/testData.csv";

    public DatasetManager(boolean pom) {
        if (pom)
            createArff();
    }

    public void createArff() {
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(",");
        Instances instances;
        try {
            loader.setSource(new File(fileName));
            instances = loader.getDataSet();
        } catch (Exception e) {
            System.err.println("Exception" + e);
            throw new RuntimeException(e);
        }
        int index = instances.numAttributes() - 1;
        instances.setClassIndex(index);

        NumericToNominal numericToNominal = new NumericToNominal();
        String s = String.valueOf(instances.numAttributes());
        numericToNominal.setAttributeIndices(s);

        Instances processedInstances;
        try {
            numericToNominal.setInputFormat(instances);
            processedInstances = Filter.useFilter(instances, numericToNominal);
        } catch (Exception e) {
            System.out.println("Exception" + e);
            throw new RuntimeException(e);
        }


        System.out.println("----------------------------------------");
        System.out.println(processedInstances);
        System.out.println("----------------------------------------");
        saveAsArff(processedInstances);
    }


    private void saveAsArff(Instances instances) {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        try {
            saver.setFile(new File("src/main/resources/files/testData.arff"));
            saver.writeBatch();
        } catch (Exception e) {
            System.out.println("saveAsArff" + e);
        }

    }
}