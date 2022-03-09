package statistics;

import lombok.AllArgsConstructor;
import lombok.Getter;
import structure.Structure;
import structure.basic.BruteForce;
import structure.trees.ballTree.BallTree;
import structure.trees.kdtree.KdTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.io.FileWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

public class Structures {

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


    private static final int randomSize = 100;


    private static final int attrSize = 50;
    private static final int k = 2;
    private static final int neighboursK = 20;
    private static final int classIndex = 0;

    private static final int instancesSize = 10_000;
    private static final int instancesSizeK = 2_000;

    public static final int ABOVE_BORDER = 1000;
    public static final int BOTTOM_BORDER = 10;

    private static Instances baseInstances;


    private static final long[][] times = new long[randomSize][timeLength];

    public static void main(String[] args) throws Exception {
        for (int i = 0; i < randomSize; i++) {
            baseInstances = new Instances("Test", getAttr(), attrSize);
            System.out.println(i);
            rand = new Random(i);
            setInstances();
            bruteForce(i);
            kdTree(i);
            ballTree(i);
            ballTreeWeka(i);
            kdTreeWeka(i);
        }
        saveTimeValues();
    }

    private static void kdTreeWeka(int i) throws Exception {
        weka.core.neighboursearch.KDTree structure = new weka.core.neighboursearch.KDTree();
        //long l = measureBuildWeka(structure);
        structure.setInstances(baseInstances);
        long l = measureNeighbours(structure);
        times[i][Time.wekaKdTree.ordinal()] = l;
    }

    private static void ballTreeWeka(int i) throws Exception {
        weka.core.neighboursearch.BallTree structure = new weka.core.neighboursearch.BallTree();
        //long l = measureBuildWeka(structure);
        structure.setInstances(baseInstances);
        long l = measureNeighbours(structure);
        times[i][Time.wekaBallTree.ordinal()] = l;

    }

    private static void bruteForce(int i) throws Exception {
        BruteForce bruteForce = new BruteForce();
        //long l = measureBuild(bruteForce);
        bruteForce.buildStructure(baseInstances);
        long l = measureNeighbours(bruteForce);
        times[i][Time.bruteForce.ordinal()] = l;
    }

    private static void kdTree(int i) throws Exception {
        KdTree kdTree = new KdTree(true);
        //long l = measureBuild(kdTree);
        kdTree.buildStructure(baseInstances);
        long l = measureNeighbours(kdTree);
        times[i][Time.kdTree.ordinal()] = l;
    }


    private static void ballTree(int i) throws Exception {
        BallTree ballTree = new BallTree(k);
        //long l = measureBuild(ballTree);
        ballTree.buildStructure(baseInstances);
        long l = measureNeighbours(ballTree);
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
                double val = BOTTOM_BORDER + (ABOVE_BORDER - BOTTOM_BORDER) * rand.nextDouble();
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
            return Double.compare(o.distance, this.distance);
        }
    }


    private static void saveTimeValues() {
        FileWriter writer;
        long[][] sum = new long[randomSize][timeLength];
        try {
            writer = new FileWriter("src/main/resources/files/statistics/stat.csv");
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
        } catch (Exception e) {
            System.out.println("Exception " + e);
        }
        for (int i = 0; i < timeLength; i++) {
            System.out.println(text[i] + sum[randomSize - 1][i]);
        }
    }
}
