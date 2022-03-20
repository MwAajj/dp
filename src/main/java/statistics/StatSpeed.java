package statistics;

import instance.InstanceManager;
import lombok.AllArgsConstructor;
import lombok.Getter;
import classifier.structure.Structure;
import classifier.structure.basic.BruteForce;
import classifier.structure.trees.ballTree.BallTree;
import classifier.structure.trees.kdtree.KdTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.io.FileWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class StatSpeed {
    enum Variable {
        attr,
        k
    }

    enum Time {
        bruteForce,
        kdTree,
        ballTree,
        wekaBallTree,
        wekaKdTree
    }

    private static final String[] text = {
            "Brute force sum: ",
            "KD structure sum: ",
            "Ball structure sum: ",
            "Weka Kd structure sum: ",
            "Weka ball structure sum: "
    };


    private static final int timeLength = 5;
    private static Random rand;


    private static final int randomSize = 10;

    private static final boolean statBuild = false;
    private static final boolean isRandom = false;


    private static final int classIndex = 0;

    private static final int instancesSize = 10_000;
    private static final int instancesSizeK = 2_000;

    private static final int ABOVE_BORDER = Integer.MAX_VALUE;
    private static final int BOTTOM_BORDER = 0;
    private static FileWriter resultFile;


    private static int attrSize;
    private static int neighboursK;
    private static String type;
    private static Instances baseInstances;

    private static long[][] times = new long[randomSize][timeLength];
    private static long[] results = new long[timeLength];

    private static final int[][] options = {
            {3, 3},
            {5, 5},
            {3, 10},
            {10, 3},
            {10, 10},
            {15, 15},
            {20, 3},
            {3, 20},
            {25, 25},
            {30, 30},
            {30, 5},
            {5, 30},
    };

    private static final String[] files =
            {
                    "Covid_I",
                    "Diabetes",
                    "EMG",
                    "Gender",
                    "IRIS",
                    "Maternal",
            };

    private static String fileName;

    public static void main(String[] args) throws Exception {
        type = "_neighbour_";
        if (statBuild)
            type = "_build_";
        resultFile = new FileWriter("src/main/resources/files/statistics/stat_results" + type + ".csv");
        if (!isRandom) {
            for (String file : files) {
                System.out.println("file: " + file);
                fileName = file;
                InstanceManager manager = new InstanceManager(fileName, classIndex);
                baseInstances = manager.getTrain();
                execute();
            }
        } else execute();
        resultFile.close();
    }

    private static void execute() throws Exception {
        for (int[] option : options) {
            times = new long[randomSize][timeLength];
            results = new long[timeLength];
            System.out.println("-------------------------------------------");
            attrSize = option[Variable.attr.ordinal()];
            neighboursK = option[Variable.k.ordinal()];
            if (!isRandom) {
                attrSize = baseInstances.firstInstance().numAttributes();
            }
            for (int i = 0; i < randomSize; i++) {
                System.out.println(i);
                rand = new Random(i);
                if (isRandom) {
                    baseInstances = new Instances("Test", getAttr(), attrSize);
                    setInstances();
                }
                bruteForce(i);
                kdTree(i);
                ballTree(i);
                ballTreeWeka(i);
                kdTreeWeka(i);
            }
            saveTimeValues();
        }
    }

    private static void kdTreeWeka(int i) throws Exception {
        weka.core.neighboursearch.KDTree structure = new weka.core.neighboursearch.KDTree();
        long l;
        if (statBuild)
            l = measureBuildWeka(structure);
        else {
            structure.setInstances(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.wekaKdTree.ordinal()] = l;
    }

    private static void ballTreeWeka(int i) throws Exception {
        weka.core.neighboursearch.BallTree structure = new weka.core.neighboursearch.BallTree();
        long l;
        if (statBuild) {
            l = measureBuildWeka(structure);
        } else {
            structure.setInstances(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.wekaBallTree.ordinal()] = l;
    }

    private static void bruteForce(int i) throws Exception {
        BruteForce structure = new BruteForce();
        long l;
        if (statBuild)
            l = measureBuild(structure);
        else {
            structure.buildStructure(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.bruteForce.ordinal()] = l;
    }

    private static void kdTree(int i) throws Exception {
        KdTree structure = new KdTree(true);
        long l;
        if (statBuild)
            l = measureBuild(structure);
        else {
            structure.buildStructure(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.kdTree.ordinal()] = l;
    }


    private static void ballTree(int i) throws Exception {
        BallTree structure = new BallTree();
        long l;
        if (statBuild)
            l = measureBuild(structure);
        else {
            structure.buildStructure(baseInstances);
            l = measureNeighbours(structure);
        }
        times[i][Time.ballTree.ordinal()] = l;
    }

    private static long measureBuildWeka(NearestNeighbourSearch structure) throws Exception {
        long start = System.currentTimeMillis();
        structure.setInstances(baseInstances);
        long finish = System.currentTimeMillis();
        return finish - start;
    }

    private static long measureBuild(Structure structure) {
        long start = System.currentTimeMillis();
        structure.buildStructure(baseInstances);
        long finish = System.currentTimeMillis();
        return finish - start;
    }

    private static long measureNeighbours(NearestNeighbourSearch structure) throws Exception {
        long start = System.currentTimeMillis();
        for (int a = 0; a < instancesSizeK; a++) {
            Instance instance = baseInstances.get(rand.nextInt(baseInstances.size()));
            structure.kNearestNeighbours(instance, neighboursK);
        }
        long finish = System.currentTimeMillis();
        return finish - start;
    }

    private static void setInstances() {
        for (int m = 0; m < instancesSize; m++) {
            double[] values = new double[attrSize];
            for (int n = 0; n < attrSize; n++) {
                int val = BOTTOM_BORDER + rand.nextInt(ABOVE_BORDER - BOTTOM_BORDER);
                values[n] = val;
            }
            baseInstances.add(new DenseInstance(1d, values));
        }
        baseInstances.setClassIndex(classIndex);
    }

    private static ArrayList<Attribute> getAttr() {
        ArrayList<Attribute> attr = new ArrayList<>(attrSize);
        for (int i = 0; i < attrSize; i++) {
            Attribute x = new Attribute(String.valueOf(i), i);
            attr.add(x);
        }
        return attr;
    }

    @AllArgsConstructor
    @Getter
    static class DistInst implements Comparable<DistInst>, Serializable {
        private Instance instance;
        private double distance;

        @Override
        public int compareTo(DistInst o) {
            return Double.compare(o.getDistance(), this.distance);
        }
    }


    private static void saveTimeValues() {
        FileWriter writer;
        long[][] sum = new long[randomSize][timeLength];
        try {

            if (isRandom)
                writer = new FileWriter("src/main/resources/files/statistics/stat" + type + attrSize + "A___" + neighboursK + "K.csv");
            else
                writer = new FileWriter("src/main/resources/files/statistics/stat_" + type + fileName + "_" + attrSize + "A___" + neighboursK + "_.csv");
            for (int i = 0; i < randomSize; i++) {
                for (int j = 0; j < times[i].length; j++) {
                    if (i == 0) sum[i][j] += times[i][j];
                    else sum[i][j] = sum[i - 1][j] + times[i][j];
                    writer.append(String.valueOf(times[i][j]));
                    writer.append(";");

                }
                for (int j = 0; j < times[i].length; j++) {
                    writer.append(String.valueOf(sum[i][j]));
                    writer.append(";");
                }
                writer.append("\n");
            }
            writer.close();

            for (int i = 0; i < results.length; i++) {
                results[i] = times[randomSize -1][i];
                resultFile.append(String.valueOf(results[i]));
                resultFile.append(";");
            }
            resultFile.append("\n");
        } catch (Exception e) {
            System.out.println("Exception " + e);
        }
        for (int i = 0; i < timeLength; i++) {
            System.out.println(text[i] + sum[randomSize - 1][i]);
        }
    }
}
