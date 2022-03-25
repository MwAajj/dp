package tests;

import classifier.EuclideanDistance;
import classifier.MyAlgorithm;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.*;

import java.util.ArrayList;
import java.util.Random;

public class VariantTest {
    private static Random rand;
    private static final int randomSize = 1000;
    private static final int attrSize = 3;
    private static final int instancesSize = 5;

    private static final int neighboursSizeK = 2;
    private static final int instancesSizeK = 2;


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
    private static String[] options = {"-K", String.valueOf(neighboursSizeK), "-H"};

    public static void main(String[] args) throws Exception {
        testHmdKnn();
        //generalTest();
    }

    private static void testHmdKnn() throws Exception {

        ArrayList<Attribute> attr = new ArrayList<>(3);
        Attribute x = new Attribute("x", 0);
        Attribute y = new Attribute("y", 1);
        Attribute z = new Attribute("z", 2);

        attr.add(x);
        attr.add(y);
        attr.add(z);

        Instances inst = new Instances("hmd", attr, 10);
        inst.add(new DenseInstance(1d, new double[]{2, 1, 0}));
        inst.add(new DenseInstance(1d, new double[]{7, 4, 0}));
        inst.add(new DenseInstance(1d, new double[]{6, 1, 0}));
        inst.add(new DenseInstance(1d, new double[]{6, 5, 1}));
        inst.add(new DenseInstance(1d, new double[]{2, 2, 0}));
        inst.add(new DenseInstance(1d, new double[]{2, 4, 0}));
        inst.add(new DenseInstance(1d, new double[]{3, 5, 1}));
        inst.add(new DenseInstance(1d, new double[]{7, 2, 1}));
        inst.setClassIndex(2);

        Instances nn = new Instances("nn", attr, 6);
        Instance target = new DenseInstance(1d, new double[]{5, 2, 1});
        nn.add(target);
        nn.setClassIndex(2);


        MyAlgorithm cc = new MyAlgorithm();
        options = new String[]{"-K", "3", "-H"};
        cc.setMDistanceFunction(new EuclideanDistance());
        cc.setOptions(options);
        cc.buildClassifier(inst);

        cc.classifyInstance(nn.firstInstance());
    }

    private static void generalTest() throws Exception {
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