package statistics;

import classifier.MyAlgorithm;
import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.neighboursearch.BallTree;

import java.io.FileWriter;
import java.io.IOException;

public class StatsPrecision {
    private static final int RANDOM_SIZE = 10;
    private static final String[] files = {
            "Covid_I",
            "Diabetes",
            "Cardio",
            "Gender",
            "IRIS",
            "Maternal"
    };
    private static final String[][] classOptions = {
            {"healthy", "ill"},
            {"healthy", "ill"},
            {"normal", "suspect", "pathologic"},
            {"male", "female"},
            {"Setosa", "Versicolor", "Virginica"},
            {"high risk", "low risk", "mid risk"}
    };

    private static final int[] kVariables = {3, 7, 11};

    public static void main(String[] args) throws Exception {
        //test();
        FileWriter resultFile = new FileWriter("src/main/resources/files/statistics/statResultPrecision.csv");
        for (int i = 0; i < files.length; i++) {
            String fileName = files[i];
            System.out.println("Files" + fileName);
            DatasetManager datasetManager = new DatasetManager(fileName, fileName, 0);

            InstanceManager manager = new InstanceManager(datasetManager.getOutputFileName(), 0);
            Instances train = manager.getTrain();
            Instances test = manager.getTest();
            Instances all = manager.getAll();
            for (int k : kVariables) {
                String newFilename;
                newFilename = fileName + "_" + k;
                String[][] myOptions = {
                        {"-K", String.valueOf(k), "-B"},
                        {"-K", String.valueOf(k), "-B", "-W"},
                        {"-K", String.valueOf(k), "-B", "-F", "2"},
                        {"-K", String.valueOf(k), "-B", "-H"},
                };

                String[][] wekaOption = {
                        {"-K", String.valueOf(k)},
                        {"-K", String.valueOf(k), "-I"},
                        {"-K", String.valueOf(k), "-F"}
                };


                double[][] sumMk = new double[RANDOM_SIZE][myOptions.length];
                double[][] sumWeka = new double[RANDOM_SIZE][wekaOption.length];


                FileWriter writer = new FileWriter("src/main/resources/files/statistics/stat" + newFilename + ".csv");
                for (int l = 0; l < RANDOM_SIZE; l++) {
                    EvaluationManager evaluation = new EvaluationManager(test, all, classOptions[i], l);
                    System.out.println("L: " + l);

                    for (int j = 0; j < myOptions.length; j++) {
                        System.out.println("\t M J:" + j);
                        MyAlgorithm alg = new MyAlgorithm();
                        String[] option = myOptions[j];
                        alg.setOptions(option);
                        alg.buildClassifier(train);

                        evaluation.evaluateModel(alg);

                        double[] infoOption = evaluation.getInfoData();
                        if (l == 0)
                            sumMk[l][j] = infoOption[infoOption.length - 1];
                        else sumMk[l][j] = sumMk[l - 1][j] + infoOption[infoOption.length - 1];
                    }

                    for (int j = 0; j < wekaOption.length; j++) {
                        System.out.println("\t W J:" + j);
                        IBk weka = new IBk();
                        weka.setNearestNeighbourSearchAlgorithm(new BallTree());
                        weka.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new EuclideanDistance());
                        String[] option = wekaOption[j];
                        weka.setOptions(option);
                        weka.buildClassifier(train);
                        evaluation.evaluateModel(weka);

                        double[] infoDataWeka = evaluation.getInfoData();
                        if (l == 0)
                            sumWeka[l][j] = infoDataWeka[infoDataWeka.length - 1];
                        else
                            sumWeka[l][j] = sumWeka[l - 1][j] + infoDataWeka[infoDataWeka.length - 1];
                    }
                    saveEvaluation(writer, sumMk[l], sumWeka[l]);
                }
                saveEvaluation(resultFile, sumMk[RANDOM_SIZE - 1], sumWeka[RANDOM_SIZE - 1]);
                writer.close();
            }
        }
        resultFile.close();
    }


    public static void saveEvaluation(FileWriter writer, double[] mk, double[] weka) throws IOException {
        for (double v : mk) {
            System.out.println("MK: " + v);
            writer.append(String.valueOf(v));
            writer.append(";");
        }
        writer.append(";");
        for (double v : weka) {
            System.out.println("WEKA: " + v);
            writer.append(String.valueOf(v));
            writer.append(";");
        }
        writer.append("\n");
    }
}
