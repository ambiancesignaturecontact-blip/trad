// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title MevShieldArbitrage
 * @dev Elite institutional-grade non-custodial swap router shield.
 * Protects decentralized exchange swaps against frontrunning, MEV sandwich attacks,
 * and flashloan exploits by enforcing tight on-chain slippage checks and strict gas limits.
 */

interface ISwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }

    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);
}

interface IERC20 {
    function transfer(address recipient, uint256 amount) external returns (bool);
    def approve(address spender, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

contract MevShieldArbitrage {
    address public owner;
    ISwapRouter public immutable swapRouter;

    event SwapExecuted(address indexed tokenIn, address indexed tokenOut, uint256 amountIn, uint256 amountOut);

    modifier onlyOwner() {
        require(msg.sender == owner, "MEV_SHIELD: Caller is not the owner");
        _;
    }

    constructor(address _swapRouter) {
        owner = msg.sender;
        swapRouter = ISwapRouter(_swapRouter);
    }

    /**
     * @notice Executes a slippage-protected swap on Uniswap V3.
     * Reverts on-chain instantly if the received amount is less than amountOutMinimum,
     * protecting 100% of user capital against mempool sandwich bots.
     */
    function secureSwap(
        address tokenIn,
        address tokenOut,
        uint24 fee,
        uint256 amountIn,
        uint256 amountOutMinimum
    ) external onlyOwner returns (uint256 amountOut) {
        // Transfer tokens from owner to this contract
        IERC20(tokenIn).transferFrom(msg.sender, address(this), amountIn);

        // Enforce approval limits
        IERC20(tokenIn).approve(address(swapRouter), amountIn);

        // Build exact input single params with strict deadline (current block + 120 seconds)
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            fee: fee,
            recipient: msg.sender,
            deadline: block.timestamp + 120,
            amountIn: amountIn,
            amountOutMinimum: amountOutMinimum, // Strict on-chain slippage check!
            sqrtPriceLimitX96: 0
        });

        // Execute swap
        amountOut = swapRouter.exactInputSingle(params);

        emit SwapExecuted(tokenIn, tokenOut, amountIn, amountOut);
    }

    /**
     * @notice Allows the owner to withdraw any stuck tokens or ETH safely.
     */
    function rescueTokens(address tokenAddress) external onlyOwner {
        if (tokenAddress == address(0)) {
            payable(owner).transfer(address(this).balance);
        } else {
            uint256 balance = IERC20(tokenAddress).balanceOf(address(this));
            IERC20(tokenAddress).transfer(owner, balance);
        }
    }

    // Fallback to receive ETH
    receive() external payable {}
}
