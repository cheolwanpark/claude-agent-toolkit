#!/usr/bin/env python3
# main.py - Backward compatibility redirect to all_demos.py

import asyncio

# Import from the new all_demos orchestrator
from examples.all_demos import main


if __name__ == "__main__":
    # Run all demos for backward compatibility
    asyncio.run(main())