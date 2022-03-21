package classifier;

import classifier.variants.*;
import classifier.variants.advanced.FuzzyKnn;
import classifier.variants.advanced.HarmonicKnn;
import classifier.variants.basic.Knn;
import classifier.variants.basic.WeightedKnn;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import classifier.structure.Structure;
import classifier.structure.basic.BruteForce;
import classifier.structure.trees.ballTree.BallTree;
import classifier.structure.trees.kdtree.KdTree;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.*;

@Getter
@Setter
@NoArgsConstructor
public class MyAlgorithm extends AbstractClassifier implements Classifier, OptionHandler {
    private Variant variant;
    private int mNumberClasses = 0;
    private boolean mkVariance = false;
    private int k = 1;
    private Structure structure;
    private DistanceFunction mDistanceFunction;

    public MyAlgorithm(int k) {
        this.k = k;
    }

    @Override
    public void buildClassifier(Instances data) {
        checkData(data);
        mNumberClasses = data.numDistinctValues(data.classIndex());
        if (mDistanceFunction == null) mDistanceFunction = new EuclideanDistance();
        if (structure == null) structure = new BallTree(k);
        structure.setDistanceFunction(mDistanceFunction);
        structure.buildStructure(data);
    }

    //return end class of new instance
    @Override
    public double classifyInstance(Instance instance) {
        return variant.classifyInstance(instance, mNumberClasses);
    }

    //return probabilities for each class of new instance
    @Override
    public double[] distributionForInstance(Instance instance) {
        return variant.distributionForInstance(instance, mNumberClasses);
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String knnString = Utils.getOption('K', options);
        String fuzzyString = Utils.getOption('F', options);

        if (knnString.length() != 0) {
            setK(Integer.parseInt(knnString));
        }
        if (Utils.getFlag('V', options))
            mkVariance = true;
        if (Utils.getFlag('B', options))
            structure = new BallTree(k);
        else if (Utils.getFlag('D', options))
            structure = new KdTree(mkVariance);
        else
            structure = new BruteForce();
        if (Utils.getFlag('H', options)) {
            variant = new HarmonicKnn(structure, k);
        } else if (fuzzyString.length() != 0) {
            variant = new FuzzyKnn(structure, k, Integer.parseInt(fuzzyString));
        } else if (Utils.getFlag('W', options)) {
            variant = new WeightedKnn(structure, k);
        } else {
            variant = new Knn(structure, k);
        }
    }

    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<>();
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

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);

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
