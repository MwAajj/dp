package tests;

import classifier.MyAlgorithm;
import instance.InstanceManager;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class VariantTest {
    private static Random rand;
    private static final int randomSize = 1000;
    private static final int attrSize = 3;
    private static final int instancesSize = 20;

    private static final int neighboursSizeK = 5;
    private static final int instancesSizeK = 5;


    private static final int classIndex = attrSize - 1;

    public static final int ABOVE_BORDER = 10;
    public static final int BOTTOM_BORDER = 0;

    public static final int ABOVE_BORDER_K = 100;
    public static final int BOTTOM_BORDER_K = 90;

    private static Instances instanceArrayList;
    private static Instances baseInstances;

    private static final int VALUE_NEIGHBOURS = 1;
    private static final int VALUE_OTHERS = 0;

    private static MyAlgorithm classifier;
    private static final String[] options = {"-K", String.valueOf(neighboursSizeK), "-B", "-W"};

    public static void main(String[] args) throws Exception {
        classifier = new MyAlgorithm();

        classifier.setOptions(options);
        for (int i = 0; i < randomSize; i++) {
            instanceArrayList = new Instances("Test", getAttr(), 2);
            baseInstances = new Instances("Test", getAttr(), 2);
            System.out.println(i);
            rand = new Random(i);
            setInstances();
            classifier.buildClassifier(baseInstances);
            testVariant();
        }
    }

    private static void testVariant() {
        for (int i = 0; i < instancesSizeK; i++) {
            Instance instance = instanceArrayList.get(i);
            double v = classifier.classifyInstance(instance);
            double[] x = classifier.distributionForInstance(instance);
            /*if(v != VALUE_NEIGHBOURS)
                throw new RuntimeException("ERROR");*/
        }
    }


    private static void setInstances() {
        for (int i = 0; i < instancesSize; i++) {
            double[] values = new double[attrSize];
            for (int j = 0; j < attrSize - 1; j++) {
                double val = rand.nextInt(((ABOVE_BORDER - BOTTOM_BORDER) + 1)) + BOTTOM_BORDER;
                values[j] = val;
            }
            values[attrSize - 1] = VALUE_OTHERS;
            baseInstances.add(new DenseInstance(1d, values));
        }
        for (int i = 0; i < instancesSizeK; i++) {
            double[] values = new double[attrSize];
            for (int j = 0; j < attrSize - 1; j++) {
                double val = rand.nextInt(((ABOVE_BORDER_K - BOTTOM_BORDER_K) + 1)) + BOTTOM_BORDER_K;
                values[j] = val;
            }
            values[attrSize - 1] = VALUE_NEIGHBOURS;
            instanceArrayList.add(new DenseInstance(1d, values));
            baseInstances.add(new DenseInstance(1d, values));
        }
        baseInstances.setClassIndex(classIndex);
        instanceArrayList.setClassIndex(classIndex);
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
