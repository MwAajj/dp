package statistics;

import lombok.AllArgsConstructor;
import lombok.Getter;
import structure.MathOperation;
import structure.Structure;
import structure.trees.ballTree.BallTree;
import structure.trees.kdtree.KdTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Structures {
    private static Random rand;
    private static final int randomSize = 10;
    private static final int attrSize = 3;
    private static final int k = 2;
    private static final int neighboursK = 3;
    private static final int classIndex = 0;

    private static final int instancesSize = 10_000;
    private static final int instancesSizeK = 2_000;

    public static final int ABOVE_BORDER = 1000;
    public static final int BOTTOM_BORDER = 10;

    private static Instances baseInstances;

    private static final long[] ballTimes = new long[randomSize];
    private static final long[] kdTimes = new long[randomSize];
    private static final long[] bruteForce = new long[randomSize];
    private static final long[] kdWeka = new long[randomSize];
    private static final long[] btWeka = new long[randomSize];

    public static void main(String[] args) {
        for (int i = 0; i < randomSize; i++) {
            baseInstances = new Instances("Test", getAttr(), attrSize);
            System.out.println(i);
            rand = new Random(i);
            setInstances();
            //ballTree(i);
            //ballTreeWeka(i);
            //kdTreeWeka(i);
            kdTree(i);
            //bruteForce(i);
        }
        //saveTimeValues();
    }

    private static void kdTreeWeka(int i) {
        try {
            weka.core.neighboursearch.KDTree structure = new weka.core.neighboursearch.KDTree();
            structure.setInstances(baseInstances);
            long start = System.currentTimeMillis();
            for (int j = 0; j < instancesSizeK; j++) {
                Instance target = baseInstances.get(rand.nextInt(baseInstances.size()));
                structure.kNearestNeighbours(target, neighboursK);
            }
            long finish = System.currentTimeMillis();
            long timeElapsed = finish - start;
            kdWeka[i] = timeElapsed;
        } catch (Exception e) {
            System.out.println("XXX");
        }
    }

    private static void ballTreeWeka(int i) {
        try {
            weka.core.neighboursearch.BallTree structure = new weka.core.neighboursearch.BallTree();
            structure.setInstances(baseInstances);
            long start = System.currentTimeMillis();
            for (int j = 0; j < instancesSizeK; j++) {
                Instance target = baseInstances.get(rand.nextInt(baseInstances.size()));
                structure.kNearestNeighbours(target, neighboursK);
            }
            long finish = System.currentTimeMillis();
            long timeElapsed = finish - start;
            btWeka[i] = timeElapsed;
        } catch (Exception e) {
            System.out.println("XXX");
        }
    }

    private static void bruteForce(int i) {
        long start = System.currentTimeMillis();
        bruteForceNeighbour();
        long finish = System.currentTimeMillis();
        long timeElapsed = finish - start;
        bruteForce[i] = timeElapsed;
    }

    private static void bruteForceNeighbour() {
        for (int j = 0; j < instancesSizeK; j++) {
            DistInst[] neighbours = new DistInst[baseInstances.size()];
            Instance target = baseInstances.get(rand.nextInt(baseInstances.size()));
            int i = 0;
            for (Instance instance : baseInstances) {
                double v = MathOperation.euclidDistance(classIndex, instance, target);
                neighbours[i] = new DistInst(instance, v);
                i++;
            }
            Arrays.sort(neighbours);
        }
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
        int sumB = 0, sumK = 0, sumBF = 0, sumWB = 0, sumWK = 0;
        try {
            writer = new FileWriter("src/main/resources/files/statistics/stat.csv");
            for (int i = 0; i < ballTimes.length; i++) {
                sumB += ballTimes[i];
                sumK += kdTimes[i];
                sumBF += bruteForce[i];
                sumWB += btWeka[i];
                sumWK += kdWeka[i];
                writer.append(String.valueOf(ballTimes[i]));
                writer.append(";");
                writer.append(String.valueOf(kdTimes[i]));
                writer.append(";");
                writer.append(String.valueOf(bruteForce[i]));
                writer.append(";");
                writer.append(String.valueOf(btWeka[i]));
                writer.append(";");
                writer.append(String.valueOf(kdWeka[i]));
                writer.append("\n");
            }
            writer.close();
        } catch (Exception e) {
            System.out.println("Exception " + e);
        }
        System.out.println("Ball structure sum: " + sumB);
        System.out.println("KD structure sum: " + sumK);
        System.out.println("Brute force sum: " + sumBF);
        System.out.println("Weka Kd structure sum: " + sumWK);
        System.out.println("Weka ball structure sum: " + sumWB);
    }

    private static void kdTree(int i) {
        KdTree kdTree = new KdTree(true);
        kdTree.buildStructure(baseInstances);
        long start = System.currentTimeMillis();
        testNeighbours(kdTree);
        long finish = System.currentTimeMillis();
        long timeElapsed = finish - start;
        kdTimes[i] = timeElapsed;
    }


    private static void ballTree(int i) {
        BallTree ballTree = new BallTree(k);
        ballTree.buildStructure(baseInstances);
        long start = System.currentTimeMillis();
        testNeighbours(ballTree);
        long finish = System.currentTimeMillis();
        long timeElapsed = finish - start;
        ballTimes[i] = timeElapsed;
    }

    private static void testNeighbours(Structure structure) {
        for (int a = 0; a < instancesSizeK; a++) {
            Instance instance = baseInstances.get(rand.nextInt(baseInstances.size()));
            structure.findKNearestNeighbours(instance, neighboursK);
        }
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

}
