package classifier;

import classifier.variants.*;
import classifier.variants.advanced.FuzzyKnn;
import classifier.variants.advanced.HarmonicKnn;
import classifier.variants.basic.Knn;
import classifier.variants.basic.WeightedKnn;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import structure.Tree;
import structure.ballTree.BallTree;
import structure.kdtree.KdTree;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.*;

@Getter
@Setter
@NoArgsConstructor
public class MyAlgorithm extends AbstractClassifier implements Classifier, OptionHandler {
    private Variant variant;
    private int m_NumClasses = 0;
    private boolean mk_variance = false;
    private int k = 1;
    private Tree tree;

    public MyAlgorithm(int k) {
        this.k = k;
    }

    @Override
    public void buildClassifier(Instances data) {
        checkData(data);
        m_NumClasses = data.numDistinctValues(data.classIndex());
        if (tree == null) tree = new BallTree(k);
        tree.buildTree(data);
    }

    //return end class of new instance
    @Override
    public double classifyInstance(Instance instance) {
        return variant.classifyInstance(instance, m_NumClasses);
    }

    //return probabilities for each class of new instance
    @Override
    public double[] distributionForInstance(Instance instance) {
        return variant.distributionForInstance(instance, m_NumClasses);
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String knnString = Utils.getOption('K', options);
        String harmonicString = Utils.getOption('H', options);
        String fuzzyString = Utils.getOption('F', options);

        if (knnString.length() != 0) {
            setK(Integer.parseInt(knnString));
        }
        if (Utils.getFlag('V', options))
            mk_variance = true;
        if (Utils.getFlag('B', options))
            tree = new BallTree(k);
        else //must be else
            tree = new KdTree(mk_variance);
        if (harmonicString.length() != 0) {
            variant = new HarmonicKnn(tree, k, Integer.parseInt(harmonicString));
        } else if (fuzzyString.length() != 0) {
            variant = new FuzzyKnn(tree, k, Integer.parseInt(fuzzyString));
        } else if (Utils.getFlag('W', options)) {
            variant = new WeightedKnn(tree, k);
        } else {
            variant = new Knn(tree, k);
        }
    }

    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<>();
        Collections.addAll(options, super.getOptions());
        options.add("-K");
        options.add("" + getK());
        options.add(variant.getOption());
        return options.toArray(new String[0]);
    }


    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);
        return result;
    }

    private void checkData(Instances data) {
        try {
            getCapabilities().testWithFail(data);
        } catch (Exception e) {
            throw new RuntimeException("Test failed: " + e);
        }
        if (data.size() < k) {
            throw new RuntimeException("K {" + k + "} is bigger then size of data{"
                    + data.size() + "}");
        }
    }
}
