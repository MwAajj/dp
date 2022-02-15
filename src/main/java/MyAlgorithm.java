import lombok.Getter;
import lombok.Setter;
import structure.MathOperation;
import structure.Tree;
import structure.balltree.BallTree;
import structure.kdtree.KdTree;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.*;

@Getter
@Setter
public class MyAlgorithm extends AbstractClassifier implements Classifier, OptionHandler {

    private boolean mk_isKdTree = false;
    private boolean mk_isBallTree = false;
    private boolean mk_noWeight = false;
    private boolean mk_harmonicWeight = false;
    private boolean mk_fuzzyWeight = false;
    private int m_NumClasses = 0;
    private int k;
    private Tree tree;

    public MyAlgorithm(int k) {
        this.k = k;
    }

    @Override
    public void buildClassifier(Instances data) {
        m_NumClasses = data.numClasses();
        if (data.size() < k) {
            System.err.println("K {" + k + "} is bigger then size of data{"
                    + data.size() + "}. You have been warned!!!");
        }
        if (mk_isBallTree)
            tree = new KdTree();
        else if (mk_isKdTree)
            tree = new KdTree();
        tree.buildTree(data);
    }

    private double noWeight(Instance instance) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        Map<Double, Integer> occurrences = getOccurrences(instances);
        int max = Integer.MIN_VALUE;
        double endClass = 0d;
        for (Map.Entry<Double, Integer> pair : occurrences.entrySet()) {
            if (max < pair.getValue()) {
                max = pair.getValue();
                endClass = pair.getKey();
            }
        }
        return endClass;
    }

    //TODO (int)current.classValue()
    private Map<Double, Integer> getOccurrences(Instances instances) {
        Map<Double, Integer> occurrences = new HashMap<>();
        for (Instance instance : instances) {
            double val = -1d;
            try {
                val = instance.classValue();
            } catch (Exception E) {
                throw new Error("Data has no class attribute!");
            }
            Integer count = occurrences.get(val);
            occurrences.put(val, count != null ? count + 1 : 1);
        }
        return occurrences;
    }

    private Set<Double> getClassValues(Instances instances) {
        Set<Double> classes = new TreeSet<>();
        for (Instance instance : instances)
            classes.add(instance.classValue());
        return classes;
    }

    private double mk_fuzzyWeight(Instance instance) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        Set<Double> classValues = getClassValues(instances);
        double endClass = 0d, min = -1d;
        Map<Double, Double> info = new HashMap<>();
        for (Double value : classValues) {
            double prob = MathOperation.fuzzyDistance(instances, instance, value, 2);
            if (min < prob) {
                min = prob;
                endClass = value;
            }
            info.put(value, prob);
        }
        System.out.println("\n--------------------F-KNN------------");
        System.out.println(info);
        System.out.println("-------------------------------------");
        return endClass;
    }


    @Override
    public double classifyInstance(Instance instance) {
        double endClass = 0d;
        if (mk_noWeight) {
            endClass = noWeight(instance);
        }
        if (mk_fuzzyWeight) {
            endClass = mk_fuzzyWeight(instance);
        }
        return endClass;
    }

    //celkovy prehlad
    @Override
    public double[] distributionForInstance(Instance instance) {
        double[] result = new double[m_NumClasses];
        Instances kNearestNeighbours = tree.findKNearestNeighbours(instance, k);
        Map<Double, Integer> occurrences = getOccurrences(kNearestNeighbours);
        for (Map.Entry<Double, Integer> pair : occurrences.entrySet()) {
            result[(int)pair.getKey().doubleValue()] = pair.getValue() / (double) k;
        }
        return result;
    }

    //kedy je ten algoritmus pouzitelny,
    // ked je to binarny klasifikator a su trie triedy exception
    // numeralne hodnoty vs lingvisticke
    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        mk_isKdTree = Utils.getFlag('K', options);
        mk_isBallTree = Utils.getFlag('B', options);

        if (Utils.getFlag('H', options))
            mk_harmonicWeight = true;
        else if (Utils.getFlag('F', options))
            mk_fuzzyWeight = true;
        else mk_noWeight = true;
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
