package tests;

import lombok.AllArgsConstructor;
import lombok.Getter;
import classifier.structure.basic.BruteForce;
import classifier.structure.trees.ballTree.BallTree;
import classifier.structure.trees.kdtree.KdTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class CompareTest {
    private static Random rand;
    private static final int randomSize = 1_000;

    private static final int attrSize = 5;
    private static final int neighboursK = 11;
    private static final int classIndex = 0;

    private static final int instancesSize = 1_000;
    private static final int instancesSizeK = 2_00;

    public static final int ABOVE_BORDER = Integer.MAX_VALUE;
    public static final int BOTTOM_BORDER = 0;

    private static Instances baseInstances;
    private static DistInst[][] bruteForceInstances = new DistInst[instancesSizeK][neighboursK];
    private static DistInst[][] kdInstances = new DistInst[instancesSizeK][neighboursK];
    private static DistInst[][] ballInstances = new DistInst[instancesSizeK][neighboursK];

    private static final double THRESHOLD = 0.000001d;

    public static void main(String[] args) throws Exception {
        for (int i = 0; i < randomSize; i++) {
            bruteForceInstances = new DistInst[instancesSizeK][neighboursK];
            kdInstances = new DistInst[instancesSizeK][neighboursK];
            ballInstances = new DistInst[instancesSizeK][neighboursK];
            baseInstances = new Instances("Test", getAttr(), attrSize);
            System.out.println(i);
            rand = new Random(i);
            setInstances();
            bruteForce();
            kdTree();
            ballTree();
            sort();
            compare();
            System.out.println();
        }
    }

    private static void sort() {
        for (int i = 0; i < instancesSizeK; i++) {
            Arrays.sort(bruteForceInstances[i], Collections.reverseOrder());
            Arrays.sort(kdInstances[i], Collections.reverseOrder());
            Arrays.sort(ballInstances[i], Collections.reverseOrder());
        }
    }

    private static void compare() {
        for (int i = 0; i < instancesSizeK; i++) {
            for (int j = 0; j < neighboursK; j++) {
                //threshold because of comparing doubles for test purposes
                if (Math.abs(bruteForceInstances[i][j].getDistance() - kdInstances[i][j].getDistance()) > THRESHOLD) {
                    throw new RuntimeException("Kd mistake");
                }
                if (Math.abs(bruteForceInstances[i][j].getDistance() - ballInstances[i][j].getDistance()) > THRESHOLD) {
                    throw new RuntimeException("Ball Mistake");
                }
            }
        }
    }

    private static void test(NearestNeighbourSearch structure, int x) throws Exception {
        for (int a = 0; a < instancesSizeK; a++) {
            Instance instance = baseInstances.get(a);
            Instances instances = structure.kNearestNeighbours(instance, neighboursK);
            double[] distances = structure.getDistances();
            addInstances(x, a, instances, distances);
        }
    }

    private static void addInstances(int x, int a, Instances instances, double[] distances) {
        for (int i = 0; i < neighboursK; i++) {
            if (x == 0)
                bruteForceInstances[a][i] = new DistInst(instances.get(i), distances[i]);
            if (x == 1)
                kdInstances[a][i] = new DistInst(instances.get(i), distances[i]);
            if (x == 2)
                ballInstances[a][i] = new DistInst(instances.get(i), distances[i]);
        }
    }

    private static void bruteForce() throws Exception {
        BruteForce structure = new BruteForce();
        structure.buildStructure(baseInstances);
        test(structure, 0);
    }

    private static void kdTree() throws Exception {
        KdTree structure = new KdTree(false);
        structure.buildStructure(baseInstances);
        test(structure, 1);
    }


    private static void ballTree() throws Exception {
        BallTree structure = new BallTree();
        structure.buildStructure(baseInstances);
        test(structure, 2);
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
            return Double.compare(o.distance, this.distance);
        }
    }
}
