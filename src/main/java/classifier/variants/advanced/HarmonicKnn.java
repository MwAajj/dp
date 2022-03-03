package classifier.variants.advanced;

import classifier.variants.Variant;
import lombok.Getter;
import lombok.Setter;
import structure.MathOperation;
import structure.Tree;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

@Getter
@Setter
public class HarmonicKnn implements Variant {
    private Tree tree;
    private int k;
    private int r;

    public HarmonicKnn(Tree tree, int k, int r) {
        if (r > k)
            throw new RuntimeException("r parameter must be smaller then k");
        this.tree = tree;
        this.k = k;
        this.r = r;
    }

    @Override
    public double[] distributionForInstance(Instance instance, int m_NumClasses) {
        return new double[1];
    }

    @Override
    public double classifyInstance(Instance instance, int m_NumClasses) {
        Instances instances = tree.findKNearestNeighbours(instance, k);
        double min = Double.MAX_VALUE, endClass = 0d;
        Map<Double, Double> info = new HashMap<>();
        for (int i = 0; i < m_NumClasses; i++) {
            double prob = MathOperation.newHarmonicDistance(instances, instance, k, r, m_NumClasses);
            if (min > prob) {
                min = prob;
                endClass = i;
            }
            info.put((double) i, prob);
        }
        System.out.println("\n--------------------HMD-KNN------------");
        System.out.println(info);
        System.out.println("-------------------------------------");
        return endClass;
    }

    @Override
    public String getOption() {
        return null;
    }
}
