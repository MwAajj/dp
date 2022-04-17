package statistics;

import classifier.MyAlgorithm;
import dataset.DatasetManager;
import evaluation.EvaluationManager;
import instance.InstanceManager;
import weka.classifiers.lazy.IBk;
import weka.core.*;
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


    private static DistanceFunction[] distanceFunctions = new DistanceFunction[]{
            new classifier.EuclideanDistance(),
            //new EuclideanDistance(),
            /*new ManhattanDistance(),
            new MinkowskiDistance(),
            new ChebyshevDistance(),*/
    };

    private static String[] distanceFunctionsNames = new String[]{
            "Mk_EuclideanDistance",
            //"EuclideanDistance",
            /*"ManhattanDistance",
            "MinkowskiDistance",
            "ChebyshevDistance",*/
    };

    private static final int[] kVariables = {3, 7, 11};

    public static void main(String[] args) throws Exception {
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
                String[][] myOptions = {
                        /*{"-K", String.valueOf(k)},
                        {"-K", String.valueOf(k), "-W"},
                        {"-K", String.valueOf(k), "-F", "2"},
                        {"-K", String.valueOf(k), "-H" , "3"},
                        {"-K", String.valueOf(k), "-D"},
                        {"-K", String.valueOf(k), "-D", "-W"},
                        {"-K", String.valueOf(k), "-D", "-F", "2"},
                        {"-K", String.valueOf(k), "-D", "-H" , "3"},*/
                        {"-K", String.valueOf(k), "-B"},
                        {"-K", String.valueOf(k), "-B", "-W"},
                        {"-K", String.valueOf(k), "-B", "-F", "2"},
                        {"-K", String.valueOf(k), "-B", "-H", "3"},
                        {"-K", String.valueOf(k), "-B", "-L", "3"},
                };

                String[][] wekaOption = {
                         {"-K", String.valueOf(k)},
                         {"-K", String.valueOf(k), "-I"},
                         {"-K", String.valueOf(k), "-F"}
                };
                String newFilename = fileName + "_" + k;
                double[][] sumMk = new double[RANDOM_SIZE][myOptions.length];
                double[][] sumWeka = new double[RANDOM_SIZE][wekaOption.length];


                for (int d = 0; d < distanceFunctions.length; d++) {
                    DistanceFunction mDistanceFunction = distanceFunctions[d];
                    System.out.println("Distance function " + distanceFunctionsNames[d]);
                    String endName = newFilename + distanceFunctionsNames[d];
                    FileWriter writer = new FileWriter("src/main/resources/files/statistics/stat" + endName + ".csv");
                    for (int l = 0; l < RANDOM_SIZE; l++) {
                        EvaluationManager evaluation = new EvaluationManager(test, all, classOptions[i], l);
                        System.out.println("L: " + l);
                        for (int j = 0; j < myOptions.length; j++) {
                            System.out.println("\t M J:" + j);
                            MyAlgorithm alg = new MyAlgorithm();
                            String[] option = myOptions[j];
                            alg.setMDistanceFunction(mDistanceFunction);
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
                            weka.getNearestNeighbourSearchAlgorithm().setDistanceFunction(mDistanceFunction);
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
