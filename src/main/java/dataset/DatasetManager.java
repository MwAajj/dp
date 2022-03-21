package dataset;


import lombok.Getter;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.File;

@Getter
public class DatasetManager {
    private static final String PATH = "src/main/resources/files/";
    private static final String CSV_SUFFIX = ".csv";
    private static final String ARF_SUFFIX = ".arff";
    private static final String DELIMITER = ",";
    private final String outputFileName;
    private final String inputFileName;
    private int index;

    public DatasetManager(String inputFileName, String outputFileName, int classIndex) {
        this.inputFileName = inputFileName;
        this.outputFileName = outputFileName;
        this.index = classIndex;
        processDataset();
    }

    public void processDataset() {
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(DELIMITER);
        Instances instances;
        String file = PATH + inputFileName + CSV_SUFFIX;
        try {
            loader.setSource(new File(file));
            instances = loader.getDataSet();
        } catch (Exception e) {
            throw new RuntimeException("Exception" + file + "_ " + e);
        }
        if (index == -1)
            index = instances.numAttributes() - 1;
        instances.setClassIndex(index);

        NumericToNominal numericToNominal = new NumericToNominal();
        String s = String.valueOf(instances.numAttributes());
        numericToNominal.setAttributeIndices(s);
        ReplaceMissingValues missing = new ReplaceMissingValues();
        Instances processedInstances;
        try {
            numericToNominal.setInputFormat(instances);
            processedInstances = Filter.useFilter(instances, numericToNominal);
            /*missing.setInputFormat(instances);
            processedInstances = Filter.useFilter(instances, missing);*/
        } catch (Exception e) {
            System.out.println("Exception" + e);
            throw new RuntimeException(e);
        }
        saveAsArff(processedInstances);
        //saveAsArff(instances);

    }


    private void saveAsArff(Instances instances) {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        try {
            String file = PATH + outputFileName + ARF_SUFFIX;
            saver.setFile(new File(file));
            saver.writeBatch();
        } catch (Exception e) {
            System.out.println("saveAsArff" + e);
        }
    }
}