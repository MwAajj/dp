import lombok.Getter;
import lombok.Setter;
import structure.Tree;
import structure.balltree.BallTree;
import structure.kdtree.KdTree;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

@Getter
@Setter
public class MyAlgorithm extends AbstractClassifier implements Classifier, OptionHandler {

    private boolean mk_isKdTree = false;
    private boolean mk_isBallTree = false;
    private boolean mk_noWeight = false;
    private boolean mk_harmonicWeight = false;
    private int mk_distanceWeighting;

    private int k;
    private Tree tree;

    public MyAlgorithm() {
        k = 1;
    }

    public MyAlgorithm(int k) {
        this.k = k;
    }


    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (mk_isBallTree)
            tree = new BallTree();
        else if (mk_isKdTree)
            tree = new KdTree();
        tree.buildTree(data);
    }

    private Map<Double, Integer> getOccurrences(Instances instances) {
        Map<Double, Integer> occurrences = new HashMap<>();
        for (Instance kNearestNeighbour : instances) {
            try {
                double val = kNearestNeighbour.classValue();
                Integer count = occurrences.get(val);
                occurrences.put(val, count != null ? count + 1 : 1);
            } catch (Exception E) {
                throw new Error("Data has no class attribute!");
            }
        }
        return occurrences;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Map<Double, Integer> occurrences = getOccurrences(tree.findKNearestNeighbours(instance, k));
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

    //celkovy prehlad
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] result = new double[instance.numAttributes()];
        Map<Double, Integer> occurrences = getOccurrences(tree.findKNearestNeighbours(instance, k));
        int index = 0;
        for (Map.Entry<Double, Integer> pair : occurrences.entrySet()) {
            result[index] = pair.getValue() / (double) k;
            index++;
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
        mk_noWeight = Utils.getFlag('N', options);
        mk_harmonicWeight = Utils.getFlag('H', options);
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
