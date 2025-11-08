# Testing Setup

## Installation

The testing dependencies are already in `package.json`. Install them with:

```bash
npm install
```

## Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode (during development)
npm run test:watch

# Run tests with coverage report
npm run test:coverage
```

## Test Structure

```
src/
└── __tests__/
    ├── physics.test.ts      # Mathematical physics validation
    ├── components.test.tsx  # React component tests
    └── hooks.test.ts        # Custom hooks tests
```

## What's Tested

### Physics Tests (`physics.test.ts`)
- **Klein-Gordon Equation**: Laplacian calculations, boundary conditions
- **Numerical Integration**: Verlet integrator energy conservation
- **Angular Momentum**: Conservation in circular orbits
- **Chi Field**: Gaussian calculations and gradients
- **Orbital Mechanics**: Circular velocity, drift detection

### Component Tests (`components.test.tsx`)
- **Error Boundary**: Catches GPU errors gracefully
- **UI Components**: Button states, accessibility

### Hook Tests (coming soon)
- **useSimulationState**: State reducer logic
- **Custom hooks**: Memoization, side effects

## Writing New Tests

### Physics Test Template
```typescript
describe('New Physics Feature', () => {
  it('should validate mathematical property', () => {
    // Arrange
    const input = ...;
    
    // Act
    const result = ...;
    
    // Assert
    expect(result).toBeCloseTo(expectedValue, precision);
  });
});
```

### Component Test Template
```typescript
describe('NewComponent', () => {
  it('should render correctly', () => {
    render(<NewComponent prop="value" />);
    expect(screen.getByText(/expected text/i)).toBeInTheDocument();
  });
});
```

## WebGPU Limitations

**Important**: WebGPU is not available in Jest/jsdom environment. Tests that require GPU:
1. Mock the GPU interface (`jest.setup.ts` already mocks `navigator.gpu`)
2. Test mathematical algorithms independently
3. Use integration tests in browser environment for GPU validation

## Coverage Goals

Current thresholds (see `jest.config.ts`):
- Branches: 50%
- Functions: 50%
- Lines: 50%
- Statements: 50%

Increase these thresholds incrementally as test coverage improves.

## Troubleshooting

### Module Resolution Errors
If you see "Cannot find module '@/...'", ensure:
1. `tsconfig.json` has path mappings configured
2. `jest.config.ts` has matching `moduleNameMapper`

### WebGPU Type Errors
If tests complain about WebGPU types:
1. Check `src/types/webgpu.d.ts` is included
2. Ensure `jest.setup.ts` mocks `navigator.gpu`

### Three.js / R3F Errors
Three.js and React Three Fiber don't work well in jsdom. For visual component tests:
1. Mock the 3D components
2. Test logic separately from rendering
3. Use Playwright/Cypress for E2E visual tests

## Next Steps

1. **Install dependencies**: `npm install`
2. **Run tests**: `npm test`
3. **Add more tests**: Focus on business logic and mathematical correctness
4. **Increase coverage**: Aim for 80% over time
5. **Setup CI/CD**: Run tests automatically on commit

## Resources

- [Jest Documentation](https://jestjs.io/)
- [React Testing Library](https://testing-library.com/react)
- [Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
